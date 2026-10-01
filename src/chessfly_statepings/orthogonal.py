from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .encoding import ACTION_INDEX, canonical_fen, encode_fen, mirror_uci


@dataclass(frozen=True, slots=True)
class OrthogonalHistory:
    current: torch.Tensor
    residue: torch.Tensor
    orthogonal: torch.Tensor
    energy_ratio: torch.Tensor
    cosine: torch.Tensor


@dataclass(frozen=True, slots=True)
class OrthogonalRow:
    fen: str
    baseline_move: str
    real_move: str
    shuffled_move: str
    real_move_changed: bool
    shuffled_move_changed: bool
    real_js_divergence: float
    shuffled_js_divergence: float
    baseline_value: float
    real_value: float
    shuffled_value: float
    real_value_delta: float
    shuffled_value_delta: float
    orthogonal_energy_ratio: float
    orthogonal_cosine: float


@dataclass(frozen=True, slots=True)
class OrthogonalRun:
    lambda_value: float
    rows: tuple[OrthogonalRow, ...]
    aggregate: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class OrthogonalResult:
    rho: float
    seed: int
    runs: tuple[OrthogonalRun, ...]


def orthogonal_component(current: torch.Tensor, residue: torch.Tensor, *, eps: float = 1e-12):
    if current.shape != residue.shape:
        raise ValueError("current and residue must have the same shape")
    if current.ndim != 2:
        raise ValueError("current and residue must have shape [batch, features]")
    h2 = torch.sum(current * current, dim=1, keepdim=True)
    dot = torch.sum(residue * current, dim=1, keepdim=True)
    coeff = torch.where(h2 > eps, dot / h2, torch.zeros_like(dot))
    orthogonal = residue - coeff * current
    residue_norm = torch.linalg.vector_norm(residue, dim=1)
    orthogonal_norm = torch.linalg.vector_norm(orthogonal, dim=1)
    current_norm = torch.linalg.vector_norm(current, dim=1)
    ratio = torch.where(residue_norm > eps, orthogonal_norm / residue_norm, torch.zeros_like(residue_norm))
    denom = current_norm * orthogonal_norm
    cosine = torch.where(
        denom > eps,
        torch.sum(current * orthogonal, dim=1) / denom,
        torch.zeros_like(denom),
    )
    return orthogonal, ratio, cosine


def orthogonal_history_readout(
    activity: Sequence[torch.Tensor],
    *,
    rho: float,
    readout_indices: torch.Tensor | Sequence[int],
) -> OrthogonalHistory:
    if not 0 <= rho < 1:
        raise ValueError("rho must satisfy 0 <= rho < 1")
    if not activity:
        raise ValueError("activity must contain at least one step")
    first = activity[0]
    if first.ndim != 2:
        raise ValueError("activity tensors must have shape [batch, neurons]")
    slow = torch.zeros_like(first)
    for hidden in activity:
        if hidden.shape != first.shape:
            raise ValueError("all activity tensors must have the same shape")
        slow = float(rho) * slow + (1.0 - float(rho)) * hidden
    residue = activity[-1] - slow
    idx = torch.as_tensor(readout_indices, dtype=torch.int64, device=first.device)
    current_readout = activity[-1].index_select(1, idx)
    residue_readout = residue.index_select(1, idx)
    orthogonal, ratio, cosine = orthogonal_component(current_readout, residue_readout)
    return OrthogonalHistory(current_readout, residue_readout, orthogonal, ratio, cosine)


def shuffled_like(values: torch.Tensor, *, seed: int) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values must have shape [batch, features]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(values.shape[1], generator=generator, device="cpu").to(values.device)
    return values.index_select(1, permutation)


def _default_board_factory(fen: str):
    try:
        import chess
    except ImportError as exc:
        raise RuntimeError("python-chess is required for orthogonal evaluation") from exc
    return chess.Board(fen)


def _decision(board: Any, mirrored: bool, policy_logits: torch.Tensor, value_logits: torch.Tensor):
    moves = tuple(board.legal_moves)
    if not moves:
        raise ValueError("cannot evaluate terminal position")
    canonical_moves = [mirror_uci(move.uci()) if mirrored else move.uci() for move in moves]
    indices = [ACTION_INDEX[uci] for uci in canonical_moves]
    logits = policy_logits[0]
    idx = torch.as_tensor(indices, dtype=torch.int64, device=logits.device)
    probs = torch.softmax(logits.index_select(0, idx), dim=0).detach().cpu().tolist()
    ranked = sorted(zip(moves, canonical_moves, probs), key=lambda item: (-float(item[2]), item[1]))
    selected_uci = ranked[0][0].uci()
    legal = {move.uci(): float(prob) for move, _canonical, prob in ranked}
    value_probs = torch.softmax(value_logits[0], dim=0).detach().cpu().tolist()
    value = sum(float(p) * ((i + 0.5) / 64.0) for i, p in enumerate(value_probs))
    return selected_uci, legal, float(value)


def _js_divergence(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    total = 0.0
    for key in sorted(set(a) | set(b)):
        p, q = float(a.get(key, 0.0)), float(b.get(key, 0.0))
        m = 0.5 * (p + q)
        if p > 0 and m > 0:
            total += 0.5 * p * math.log(p / m)
        if q > 0 and m > 0:
            total += 0.5 * q * math.log(q / m)
    return total


def evaluate_orthogonal_positions(
    fens: Iterable[str],
    model: Any,
    *,
    rho: float = 0.75,
    lambdas: Sequence[float] = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0),
    seed: int = 0,
    board_factory: Callable[[str], Any] | None = None,
) -> OrthogonalResult:
    factory = board_factory or _default_board_factory
    prepared = []
    for position_index, fen in enumerate(fens):
        board = factory(fen)
        canonical, mirrored = canonical_fen(board.fen())
        features = torch.tensor(encode_fen(canonical), dtype=torch.float32, device=model.device)
        baseline = model.forward(features, include_activity=True)
        if baseline.instability is not None or baseline.policy_logits is None or baseline.value_logits is None:
            raise RuntimeError(f"baseline instability at position {position_index}: {baseline.instability}")
        history = orthogonal_history_readout(
            baseline.activity,
            rho=rho,
            readout_indices=model.readout_index,
        )
        shuffled = shuffled_like(history.orthogonal, seed=int(seed) + position_index)
        baseline_move, baseline_probs, baseline_value = _decision(
            board,
            mirrored,
            baseline.policy_logits,
            baseline.value_logits,
        )
        prepared.append(
            (
                fen,
                board,
                mirrored,
                history,
                shuffled,
                baseline_move,
                baseline_probs,
                baseline_value,
            )
        )

    runs: list[OrthogonalRun] = []
    for lambda_value in lambdas:
        rows: list[OrthogonalRow] = []
        for fen, board, mirrored, history, shuffled, baseline_move, baseline_probs, baseline_value in prepared:
            real = model.decode_readout(history.current + float(lambda_value) * history.orthogonal)
            control = model.decode_readout(history.current + float(lambda_value) * shuffled)
            real_move, real_probs, real_value = _decision(board, mirrored, real.policy_logits, real.value_logits)
            shuffled_move, shuffled_probs, shuffled_value = _decision(
                board,
                mirrored,
                control.policy_logits,
                control.value_logits,
            )
            rows.append(
                OrthogonalRow(
                    fen=fen,
                    baseline_move=baseline_move,
                    real_move=real_move,
                    shuffled_move=shuffled_move,
                    real_move_changed=real_move != baseline_move,
                    shuffled_move_changed=shuffled_move != baseline_move,
                    real_js_divergence=_js_divergence(baseline_probs, real_probs),
                    shuffled_js_divergence=_js_divergence(baseline_probs, shuffled_probs),
                    baseline_value=baseline_value,
                    real_value=real_value,
                    shuffled_value=shuffled_value,
                    real_value_delta=real_value - baseline_value,
                    shuffled_value_delta=shuffled_value - baseline_value,
                    orthogonal_energy_ratio=float(history.energy_ratio[0].item()),
                    orthogonal_cosine=float(history.cosine[0].item()),
                )
            )
        n = len(rows)
        if n == 0:
            aggregate = {
                key: 0.0
                for key in (
                    "positions",
                    "mean_orthogonal_energy_ratio",
                    "mean_abs_orthogonal_cosine",
                    "real_move_change_rate",
                    "shuffled_move_change_rate",
                    "mean_real_js_divergence",
                    "mean_shuffled_js_divergence",
                    "mean_real_abs_value_delta",
                    "mean_shuffled_abs_value_delta",
                )
            }
        else:
            aggregate = {
                "positions": float(n),
                "mean_orthogonal_energy_ratio": sum(r.orthogonal_energy_ratio for r in rows) / n,
                "mean_abs_orthogonal_cosine": sum(abs(r.orthogonal_cosine) for r in rows) / n,
                "real_move_change_rate": sum(r.real_move_changed for r in rows) / n,
                "shuffled_move_change_rate": sum(r.shuffled_move_changed for r in rows) / n,
                "mean_real_js_divergence": sum(r.real_js_divergence for r in rows) / n,
                "mean_shuffled_js_divergence": sum(r.shuffled_js_divergence for r in rows) / n,
                "mean_real_abs_value_delta": sum(abs(r.real_value_delta) for r in rows) / n,
                "mean_shuffled_abs_value_delta": sum(abs(r.shuffled_value_delta) for r in rows) / n,
            }
        aggregate["real_minus_shuffled_js"] = (
            aggregate["mean_real_js_divergence"] - aggregate["mean_shuffled_js_divergence"]
        )
        runs.append(OrthogonalRun(float(lambda_value), tuple(rows), aggregate))
    return OrthogonalResult(float(rho), int(seed), tuple(runs))


__all__ = [
    "OrthogonalHistory",
    "OrthogonalResult",
    "OrthogonalRow",
    "OrthogonalRun",
    "evaluate_orthogonal_positions",
    "orthogonal_component",
    "orthogonal_history_readout",
    "shuffled_like",
]
