from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence

import torch

from .curvature import even_receiver_response
from .encoding import canonical_fen, encode_fen
from .orthogonal import orthogonal_component, shuffled_orthogonal_control
from .retrieval_diagnostics import center_rows


@dataclass(frozen=True, slots=True)
class FeedbackForward:
    readout: torch.Tensor
    policy_logits: torch.Tensor | None
    value_logits: torch.Tensor | None
    activity: tuple[torch.Tensor, ...]
    feedback_norms: tuple[float, ...]
    instability: str | None = None


@dataclass(frozen=True, slots=True)
class FeedbackRow:
    index: int
    readout_displacement: float
    association_displacement: float
    value_centered_rms: float
    policy_centered_rms: float
    total_feedback_norm: float


@dataclass(frozen=True, slots=True)
class FeedbackSetting:
    magnitude: float
    gamma: float
    mode: str
    mean_readout_displacement: float
    mean_association_displacement: float
    mean_value_centered_rms: float
    mean_policy_centered_rms: float
    mean_total_feedback_norm: float
    rows: tuple[FeedbackRow, ...]


@dataclass(frozen=True, slots=True)
class FeedbackDriftResult:
    rho: float
    positions: int
    seed: int
    settings: tuple[FeedbackSetting, ...]


def run_rectified_feedback(
    model,
    features,
    *,
    rho: float = 0.75,
    magnitude: float = 1.0,
    gamma: float = 0.01,
    mode: str = "nonlinear_even",
    seed: int = 0,
    include_activity: bool = False,
) -> FeedbackForward:
    """Inject an adjoint-mapped even response between settling steps.

    `nonlinear_even` uses the real orthogonal history ping. `shuffled_even`
    substitutes a norm-matched shuffled ping. `linear_even` is the explicit
    zero-curvature control and therefore injects no feedback. The feedback map
    is the frozen decoder transpose; it is an experimental tied/adjoint path,
    not part of published ChessFly.
    """
    if not 0 <= rho < 1:
        raise ValueError("rho must satisfy 0 <= rho < 1")
    magnitude = float(magnitude)
    gamma = float(gamma)
    if magnitude <= 0:
        raise ValueError("magnitude must be positive")
    if mode not in {"nonlinear_even", "shuffled_even", "linear_even"}:
        raise ValueError("unknown feedback mode")

    with torch.inference_mode():
        _, drive = model._prepare(features)
        hidden = torch.zeros_like(drive)
        slow = torch.zeros_like(drive)
        activity: list[torch.Tensor] = []
        feedback_norms: list[float] = []

        for step in range(model.weights.steps):
            recurrent = torch.sparse.mm(
                model.matrix, hidden.transpose(0, 1)
            ).transpose(0, 1)
            pre = (
                (recurrent + drive) * model.tensors["scale"][step]
                + model.tensors["shift"][step]
            )
            hidden = (
                (1.0 - model.weights.alpha) * hidden
                + model.weights.alpha * torch.relu(pre)
            )
            if not torch.isfinite(hidden).all():
                readout = hidden.index_select(1, model.readout_index)
                return FeedbackForward(
                    readout=readout,
                    policy_logits=None,
                    value_logits=None,
                    activity=tuple(activity),
                    feedback_norms=tuple(feedback_norms),
                    instability=f"non-finite activity at step {step + 1}",
                )

            slow = float(rho) * slow + (1.0 - float(rho)) * hidden
            residue = hidden - slow
            current_readout = hidden.index_select(1, model.readout_index)
            residue_readout = residue.index_select(1, model.readout_index)
            direction, _ratio, _cosine = orthogonal_component(
                current_readout, residue_readout
            )
            if mode == "shuffled_even":
                direction = shuffled_orthogonal_control(
                    current_readout,
                    direction,
                    seed=int(seed) + step,
                )

            feedback_readout = torch.zeros_like(current_readout)
            if (
                step < model.weights.steps - 1
                and gamma != 0.0
                and mode != "linear_even"
            ):
                even = even_receiver_response(
                    model,
                    current_readout,
                    direction,
                    magnitude=magnitude,
                )
                feedback_readout = even @ model.tensors["decoder.weight"]
                hidden = hidden.clone()
                hidden.index_copy_(
                    1,
                    model.readout_index,
                    current_readout + gamma * feedback_readout,
                )

            feedback_norms.append(
                float(torch.linalg.vector_norm(feedback_readout).item())
            )
            if include_activity:
                activity.append(hidden.clone())

        readout = hidden.index_select(1, model.readout_index)
        trace = model.decode_readout_trace(readout)
        return FeedbackForward(
            readout=readout,
            policy_logits=trace.policy_logits,
            value_logits=trace.value_logits,
            activity=tuple(activity),
            feedback_norms=tuple(feedback_norms),
            instability=None,
        )


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values.float() * values.float())).item())


def evaluate_feedback_drift(
    fens: Iterable[str],
    model,
    *,
    rho: float = 0.75,
    magnitudes: Sequence[float] = (0.5, 1.0, 2.0),
    gammas: Sequence[float] = (0.0, 0.01, 0.05),
    seed: int = 0,
) -> FeedbackDriftResult:
    """Compare recurrent drift from real, shuffled, and zero-curvature feedback."""
    fen_list = tuple(fens)
    if not fen_list:
        raise ValueError("feedback-drift requires at least one position")
    magnitude_values = tuple(float(value) for value in magnitudes)
    gamma_values = tuple(float(value) for value in gammas)
    if not magnitude_values or any(value <= 0 for value in magnitude_values):
        raise ValueError("magnitudes must be positive")
    if not gamma_values:
        raise ValueError("at least one gamma is required")

    prepared = []
    for index, fen in enumerate(fen_list):
        canonical, _mirrored = canonical_fen(fen)
        features = torch.tensor(
            encode_fen(canonical), dtype=torch.float32, device=model.device
        )
        baseline = model.forward(features, include_activity=True)
        if getattr(baseline, "instability", None) is not None:
            raise RuntimeError(f"baseline instability at position {index}: {baseline.instability}")
        baseline_readout = baseline.activity[-1].index_select(1, model.readout_index)
        baseline_trace = model.decode_readout_trace(baseline_readout)
        prepared.append((features, baseline_readout, baseline_trace))

    settings = []
    for magnitude in magnitude_values:
        for gamma in gamma_values:
            for mode in ("nonlinear_even", "shuffled_even", "linear_even"):
                rows = []
                for index, (features, baseline_readout, baseline_trace) in enumerate(prepared):
                    result = run_rectified_feedback(
                        model,
                        features,
                        rho=rho,
                        magnitude=magnitude,
                        gamma=gamma,
                        mode=mode,
                        seed=int(seed) + 1009 * index,
                    )
                    if result.instability is not None:
                        raise RuntimeError(
                            f"feedback instability at position {index}: {result.instability}"
                        )
                    trace = model.decode_readout_trace(result.readout)
                    rows.append(
                        FeedbackRow(
                            index=index,
                            readout_displacement=float(
                                torch.linalg.vector_norm(
                                    result.readout - baseline_readout
                                ).item()
                            ),
                            association_displacement=float(
                                torch.linalg.vector_norm(
                                    trace.association - baseline_trace.association
                                ).item()
                            ),
                            value_centered_rms=_rms(
                                center_rows(
                                    trace.value_logits - baseline_trace.value_logits
                                )
                            ),
                            policy_centered_rms=_rms(
                                center_rows(
                                    trace.policy_logits - baseline_trace.policy_logits
                                )
                            ),
                            total_feedback_norm=sum(result.feedback_norms),
                        )
                    )
                row_tuple = tuple(rows)
                settings.append(
                    FeedbackSetting(
                        magnitude=magnitude,
                        gamma=gamma,
                        mode=mode,
                        mean_readout_displacement=mean(
                            row.readout_displacement for row in row_tuple
                        ),
                        mean_association_displacement=mean(
                            row.association_displacement for row in row_tuple
                        ),
                        mean_value_centered_rms=mean(
                            row.value_centered_rms for row in row_tuple
                        ),
                        mean_policy_centered_rms=mean(
                            row.policy_centered_rms for row in row_tuple
                        ),
                        mean_total_feedback_norm=mean(
                            row.total_feedback_norm for row in row_tuple
                        ),
                        rows=row_tuple,
                    )
                )

    return FeedbackDriftResult(
        rho=float(rho),
        positions=len(fen_list),
        seed=int(seed),
        settings=tuple(settings),
    )


__all__ = [
    "FeedbackDriftResult",
    "FeedbackForward",
    "FeedbackRow",
    "FeedbackSetting",
    "evaluate_feedback_drift",
    "run_rectified_feedback",
]
