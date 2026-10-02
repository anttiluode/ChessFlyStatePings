from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
from typing import Iterable, Sequence

import torch

from .encoding import canonical_fen, encode_fen
from .listener_replacement import apply_listener
from .orthogonal import orthogonal_history_readout
from .retrieval_diagnostics import center_rows


@dataclass(frozen=True, slots=True)
class CurvatureRow:
    index: int
    even_norm: float
    scaled_even_norm: float
    analytic_norm: float
    analytic_cosine: float
    relative_rmse: float
    value_centered_norm: float
    policy_centered_norm: float


@dataclass(frozen=True, slots=True)
class CurvatureRun:
    magnitude: float
    mean_even_norm: float
    mean_scaled_even_norm: float
    mean_analytic_cosine: float
    mean_relative_rmse: float
    rows: tuple[CurvatureRow, ...]


@dataclass(frozen=True, slots=True)
class CurvatureResult:
    rho: float
    positions: int
    runs: tuple[CurvatureRun, ...]


def even_receiver_response(
    model,
    current: torch.Tensor,
    direction: torch.Tensor,
    *,
    magnitude: float,
) -> torch.Tensor:
    """Even response: balanced +/- pings after the nonlinear association decoder."""
    if current.shape != direction.shape:
        raise ValueError("current and direction must share shape")
    if current.ndim != 2:
        raise ValueError("current and direction must be matrices")
    magnitude = float(magnitude)
    if magnitude <= 0:
        raise ValueError("magnitude must be positive")
    base = model.decode_readout_trace(current).association
    plus = model.decode_readout_trace(current + magnitude * direction).association
    minus = model.decode_readout_trace(current - magnitude * direction).association
    return 0.5 * (plus + minus) - base


def gelu_tanh_second(preactivation: torch.Tensor) -> torch.Tensor:
    """Second derivative of PyTorch's tanh-approximate GELU."""
    k = math.sqrt(2.0 / math.pi)
    cubic = 0.044715
    q = k * (preactivation + cubic * preactivation.pow(3))
    q_prime = k * (1.0 + 3.0 * cubic * preactivation.pow(2))
    q_second = k * (6.0 * cubic * preactivation)
    tanh_q = torch.tanh(q)
    sech2 = 1.0 - tanh_q.pow(2)
    tanh_prime = sech2 * q_prime
    tanh_second = sech2 * (q_second - 2.0 * tanh_q * q_prime.pow(2))
    return tanh_prime + 0.5 * preactivation * tanh_second


def analytic_gelu_tanh_even_response(
    current: torch.Tensor,
    direction: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    *,
    magnitude: float,
) -> torch.Tensor:
    """Leading O(a^2) curvature prediction for the association decoder."""
    if current.shape != direction.shape:
        raise ValueError("current and direction must share shape")
    magnitude = float(magnitude)
    if magnitude <= 0:
        raise ValueError("magnitude must be positive")
    preactivation = current @ decoder_weight.T + decoder_bias
    directional_pre = direction @ decoder_weight.T
    return (
        0.5
        * magnitude
        * magnitude
        * gelu_tanh_second(preactivation)
        * directional_pre.pow(2)
    )


def _row_cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    numerator = torch.sum(left * right, dim=1)
    denominator = torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(right, dim=1)
    return numerator / torch.clamp(denominator, min=eps)


def evaluate_curvature(
    fens: Iterable[str],
    model,
    *,
    rho: float = 0.75,
    magnitudes: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> CurvatureResult:
    """Measure nonlinear rectification of balanced history pings at the final settle state."""
    fen_list = tuple(fens)
    if not fen_list:
        raise ValueError("curvature requires at least one position")
    magnitude_values = tuple(float(value) for value in magnitudes)
    if not magnitude_values or any(value <= 0 for value in magnitude_values):
        raise ValueError("magnitudes must be positive")

    currents = []
    directions = []
    for index, fen in enumerate(fen_list):
        canonical, _mirrored = canonical_fen(fen)
        features = torch.tensor(
            encode_fen(canonical), dtype=torch.float32, device=model.device
        )
        baseline = model.forward(features, include_activity=True)
        if getattr(baseline, "instability", None) is not None:
            raise RuntimeError(f"baseline instability at position {index}: {baseline.instability}")
        history = orthogonal_history_readout(
            tuple(baseline.activity), rho=rho, readout_indices=model.readout_index
        )
        currents.append(history.current)
        directions.append(history.orthogonal)

    current = torch.cat(currents, dim=0)
    direction = torch.cat(directions, dim=0)
    decoder_weight = model.tensors["decoder.weight"]
    decoder_bias = model.tensors["decoder.bias"]
    value_weight = model.tensors["value.weight"]
    policy_weight = model.tensors["policy.weight"]

    runs = []
    for magnitude in magnitude_values:
        even = even_receiver_response(model, current, direction, magnitude=magnitude)
        analytic = analytic_gelu_tanh_even_response(
            current,
            direction,
            decoder_weight,
            decoder_bias,
            magnitude=magnitude,
        )
        even_norm = torch.linalg.vector_norm(even, dim=1)
        analytic_norm = torch.linalg.vector_norm(analytic, dim=1)
        cosine = _row_cosine(even, analytic)
        relative_rmse = torch.linalg.vector_norm(even - analytic, dim=1) / torch.clamp(
            analytic_norm, min=1e-12
        )
        value_norm = torch.linalg.vector_norm(
            center_rows(apply_listener(even, value_weight)), dim=1
        )
        policy_norm = torch.linalg.vector_norm(
            center_rows(apply_listener(even, policy_weight)), dim=1
        )

        rows = tuple(
            CurvatureRow(
                index=index,
                even_norm=float(even_norm[index].item()),
                scaled_even_norm=float((even_norm[index] / (magnitude * magnitude)).item()),
                analytic_norm=float(analytic_norm[index].item()),
                analytic_cosine=float(cosine[index].item()),
                relative_rmse=float(relative_rmse[index].item()),
                value_centered_norm=float(value_norm[index].item()),
                policy_centered_norm=float(policy_norm[index].item()),
            )
            for index in range(len(fen_list))
        )
        runs.append(
            CurvatureRun(
                magnitude=magnitude,
                mean_even_norm=mean(row.even_norm for row in rows),
                mean_scaled_even_norm=mean(row.scaled_even_norm for row in rows),
                mean_analytic_cosine=mean(row.analytic_cosine for row in rows),
                mean_relative_rmse=mean(row.relative_rmse for row in rows),
                rows=rows,
            )
        )

    return CurvatureResult(
        rho=float(rho),
        positions=len(fen_list),
        runs=tuple(runs),
    )


__all__ = [
    "CurvatureResult",
    "CurvatureRow",
    "CurvatureRun",
    "analytic_gelu_tanh_even_response",
    "evaluate_curvature",
    "even_receiver_response",
    "gelu_tanh_second",
]
