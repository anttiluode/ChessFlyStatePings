from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .encoding import ACTION_INDEX, canonical_fen, encode_fen, mirror_uci
from .orthogonal import orthogonal_history_readout, shuffled_orthogonal_control

METRIC_KEYS = (
    "policy_logit_rms",
    "margin_abs_delta",
    "value_logit_rms",
    "association_rel_change",
    "js_divergence",
)


@dataclass(frozen=True, slots=True)
class SignedMetrics:
    policy_logit_rms: float
    margin_abs_delta: float
    value_logit_rms: float
    association_rel_change: float
    js_divergence: float
    move_changed: bool
    value_bin: int
    value_bin_changed: bool


@dataclass(frozen=True, slots=True)
class SpecificityRow:
    fen: str
    magnitude: float
    orthogonal_norm_ratio: float
    orthogonal_energy_fraction: float
    orthogonal_cosine: float
    real_plus: SignedMetrics
    real_minus: SignedMetrics
    real_symmetric: Mapping[str, float]
    real_percentiles: Mapping[str, float]
    control_mean: Mapping[str, float]
    control_p95: Mapping[str, float]
    control_max: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class SpecificityRun:
    magnitude: float
    rows: tuple[SpecificityRow, ...]
    aggregate: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class SpecificityResult:
    rho: float
    seed: int
    controls: int
    runs: tuple[SpecificityRun, ...]


def centered_rms_delta(base: torch.Tensor, perturbed: torch.Tensor) -> float:
    """RMS logit change after removing the irrelevant common-shift direction."""
    base = base.float().reshape(-1)
    perturbed = perturbed.float().reshape(-1)
    if base.numel() != perturbed.numel():
        raise ValueError("base and perturbed must have the same number of elements")
    if base.numel() == 0:
        return 0.0
    raw_delta = perturbed - base
    delta = raw_delta - raw_delta.mean()
    return float(torch.sqrt(torch.mean(delta * delta)).item())


def sign_symmetric(plus: float, minus: float) -> float:
    """Remove the sign cherry-pick by retaining the larger +/- effect."""
    return max(float(plus), float(minus))


def empirical_percentile(value: float, controls: Sequence[float]) -> float:
    """Mid-rank percentile with a half pseudo-observation at each edge."""
    if not controls:
        raise ValueError("controls must not be empty")
    less = sum(float(control) < float(value) for control in controls)
    equal = sum(float(control) == float(value) for control in controls)
    return (less + 0.5 * equal + 0.5) / (len(controls) + 1.0)


def _quantile95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _default_board_factory(fen: str):
    try:
        import chess
    except ImportError as exc:
        raise RuntimeError("python-chess is required for specificity evaluation") from exc
    return chess.Board(fen)


def _legal_indices(board: Any, mirrored: bool, device: torch.device) -> torch.Tensor:
    moves = tuple(board.legal_moves)
    if not moves:
        raise ValueError("cannot evaluate terminal position")
    canonical_moves = [mirror_uci(move.uci()) if mirrored else move.uci() for move in moves]
    return torch.as_tensor([ACTION_INDEX[uci] for uci in canonical_moves], dtype=torch.int64, device=device)


def _js_from_logits(base: torch.Tensor, perturbed: torch.Tensor) -> float:
    p = torch.softmax(base.float(), dim=0)
    q = torch.softmax(perturbed.float(), dim=0)
    m = 0.5 * (p + q)
    terms = torch.zeros_like(m)
    mask = p > 0
    terms[mask] += 0.5 * p[mask] * torch.log(p[mask] / m[mask])
    mask = q > 0
    terms[mask] += 0.5 * q[mask] * torch.log(q[mask] / m[mask])
    return float(terms.sum().item())


def _margin(logits: torch.Tensor) -> float:
    values = logits.float().reshape(-1)
    if values.numel() < 2:
        return 0.0
    top = torch.topk(values, k=2).values
    return float((top[0] - top[1]).item())


def _signed_metrics(base_trace: Any, trace: Any, row: int, legal_idx: torch.Tensor) -> SignedMetrics:
    base_policy = base_trace.policy_logits[0].index_select(0, legal_idx)
    policy = trace.policy_logits[row].index_select(0, legal_idx)
    base_value = base_trace.value_logits[0]
    value = trace.value_logits[row]
    base_assoc = base_trace.association[0].float()
    association = trace.association[row].float()
    denom = float(torch.linalg.vector_norm(base_assoc).item())
    association_delta = float(torch.linalg.vector_norm(association - base_assoc).item())
    association_rel_change = association_delta if denom == 0.0 else association_delta / denom
    base_move = int(torch.argmax(base_policy).item())
    move = int(torch.argmax(policy).item())
    base_bin = int(torch.argmax(base_value).item())
    value_bin = int(torch.argmax(value).item())
    return SignedMetrics(
        policy_logit_rms=centered_rms_delta(base_policy, policy),
        margin_abs_delta=abs(_margin(policy) - _margin(base_policy)),
        value_logit_rms=centered_rms_delta(base_value, value),
        association_rel_change=association_rel_change,
        js_divergence=_js_from_logits(base_policy, policy),
        move_changed=move != base_move,
        value_bin=value_bin,
        value_bin_changed=value_bin != base_bin,
    )


def _symmetric(plus: SignedMetrics, minus: SignedMetrics) -> dict[str, float]:
    return {key: sign_symmetric(getattr(plus, key), getattr(minus, key)) for key in METRIC_KEYS}


def _control_summary(control_symmetric: Sequence[Mapping[str, float]], real_symmetric: Mapping[str, float]):
    means: dict[str, float] = {}
    p95: dict[str, float] = {}
    maxima: dict[str, float] = {}
    percentiles: dict[str, float] = {}
    for key in METRIC_KEYS:
        values = [float(item[key]) for item in control_symmetric]
        means[key] = mean(values)
        p95[key] = _quantile95(values)
        maxima[key] = max(values)
        percentiles[key] = empirical_percentile(float(real_symmetric[key]), values)
    return means, p95, maxima, percentiles


def evaluate_directional_specificity(
    fens: Iterable[str],
    model: Any,
    *,
    rho: float = 0.75,
    magnitudes: Sequence[float] = (1.0, 2.0, 4.0),
    controls: int = 32,
    seed: int = 0,
    board_factory: Callable[[str], Any] | None = None,
) -> SpecificityResult:
    """Compare the real transverse history direction to many matched controls.

    Each position gets one expensive recurrent ChessFly forward. For each magnitude,
    all real/control directions and both signs are decoded in one batch.
    """
    if controls < 1:
        raise ValueError("controls must be at least 1")
    magnitude_values = tuple(float(value) for value in magnitudes)
    if not magnitude_values or any(value <= 0 or not math.isfinite(value) for value in magnitude_values):
        raise ValueError("magnitudes must be finite positive values")
    factory = board_factory or _default_board_factory
    rows_by_run: list[list[SpecificityRow]] = [[] for _ in magnitude_values]

    for position_index, fen in enumerate(fens):
        board = factory(fen)
        canonical, mirrored = canonical_fen(board.fen())
        features = torch.tensor(encode_fen(canonical), dtype=torch.float32, device=model.device)
        baseline = model.forward(features, include_activity=True)
        if baseline.instability is not None or baseline.policy_logits is None or baseline.value_logits is None:
            raise RuntimeError(f"baseline instability at position {position_index}: {baseline.instability}")
        history = orthogonal_history_readout(baseline.activity, rho=rho, readout_indices=model.readout_index)
        base_trace = model.decode_readout_trace(history.current)
        legal_idx = _legal_indices(board, mirrored, base_trace.policy_logits.device)

        control_vectors = [
            shuffled_orthogonal_control(
                history.current,
                history.orthogonal,
                seed=int(seed) + position_index * 1_000_003 + control_index + 1,
            )
            for control_index in range(controls)
        ]
        directions = torch.cat([history.orthogonal, *control_vectors], dim=0)
        expanded_current = history.current.expand(directions.shape[0], -1)

        for run_index, magnitude in enumerate(magnitude_values):
            plus = expanded_current + magnitude * directions
            minus = expanded_current - magnitude * directions
            trace = model.decode_readout_trace(torch.cat([plus, minus], dim=0))
            count = directions.shape[0]
            plus_metrics = [_signed_metrics(base_trace, trace, index, legal_idx) for index in range(count)]
            minus_metrics = [_signed_metrics(base_trace, trace, index + count, legal_idx) for index in range(count)]
            real_plus, real_minus = plus_metrics[0], minus_metrics[0]
            real_symmetric = _symmetric(real_plus, real_minus)
            control_symmetric = [_symmetric(plus_item, minus_item) for plus_item, minus_item in zip(plus_metrics[1:], minus_metrics[1:])]
            control_mean, control_p95, control_max, percentiles = _control_summary(control_symmetric, real_symmetric)
            rows_by_run[run_index].append(
                SpecificityRow(
                    fen=fen,
                    magnitude=magnitude,
                    orthogonal_norm_ratio=float(history.norm_ratio[0].item()),
                    orthogonal_energy_fraction=float(history.energy_fraction[0].item()),
                    orthogonal_cosine=float(history.cosine[0].item()),
                    real_plus=real_plus,
                    real_minus=real_minus,
                    real_symmetric=real_symmetric,
                    real_percentiles=percentiles,
                    control_mean=control_mean,
                    control_p95=control_p95,
                    control_max=control_max,
                )
            )

    runs: list[SpecificityRun] = []
    for magnitude, row_list in zip(magnitude_values, rows_by_run):
        rows = tuple(row_list)
        n = len(rows)
        aggregate: dict[str, float] = {"positions": float(n)}
        if n == 0:
            aggregate.update({
                "mean_orthogonal_norm_ratio": 0.0,
                "mean_orthogonal_energy_fraction": 0.0,
                "mean_abs_orthogonal_cosine": 0.0,
            })
            for key in METRIC_KEYS:
                aggregate[f"mean_real_{key}"] = 0.0
                aggregate[f"mean_control_{key}"] = 0.0
                aggregate[f"mean_real_percentile_{key}"] = 0.0
                aggregate[f"fraction_real_percentile_ge_0_95_{key}"] = 0.0
        else:
            aggregate.update({
                "mean_orthogonal_norm_ratio": mean(row.orthogonal_norm_ratio for row in rows),
                "mean_orthogonal_energy_fraction": mean(row.orthogonal_energy_fraction for row in rows),
                "mean_abs_orthogonal_cosine": mean(abs(row.orthogonal_cosine) for row in rows),
            })
            for key in METRIC_KEYS:
                aggregate[f"mean_real_{key}"] = mean(float(row.real_symmetric[key]) for row in rows)
                aggregate[f"mean_control_{key}"] = mean(float(row.control_mean[key]) for row in rows)
                aggregate[f"mean_real_percentile_{key}"] = mean(float(row.real_percentiles[key]) for row in rows)
                aggregate[f"fraction_real_percentile_ge_0_95_{key}"] = sum(float(row.real_percentiles[key]) >= 0.95 for row in rows) / n
        runs.append(SpecificityRun(magnitude, rows, aggregate))
    return SpecificityResult(float(rho), int(seed), int(controls), tuple(runs))


__all__ = [
    "METRIC_KEYS",
    "SignedMetrics",
    "SpecificityResult",
    "SpecificityRow",
    "SpecificityRun",
    "centered_rms_delta",
    "empirical_percentile",
    "evaluate_directional_specificity",
    "sign_symmetric",
]
