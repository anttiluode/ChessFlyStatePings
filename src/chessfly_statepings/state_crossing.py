from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence

import torch

from .encoding import canonical_fen, encode_fen
from .listener_replacement import apply_listener
from .orthogonal import orthogonal_history_readout, shuffled_orthogonal_control
from .retrieval_diagnostics import center_rows


@dataclass(frozen=True, slots=True)
class StateCrossMetric:
    alignment_matrix: tuple[tuple[float, ...], ...]
    diagonal_mean: float
    off_diagonal_mean: float
    diagonal_advantage: float
    control_advantage_mean: float
    control_advantage_percentile: float
    control_advantages: tuple[float, ...]
    linear_alignment_matrix: tuple[tuple[float, ...], ...]
    linear_diagonal_advantage: float


@dataclass(frozen=True, slots=True)
class StateCrossRun:
    magnitude: float
    association: StateCrossMetric
    value: StateCrossMetric
    policy: StateCrossMetric


@dataclass(frozen=True, slots=True)
class StateCrossResult:
    rho: float
    positions: int
    controls: int
    seed: int
    runs: tuple[StateCrossRun, ...]


def cross_odd_responses(
    model,
    states: torch.Tensor,
    pings: torch.Tensor,
    *,
    magnitude: float,
) -> torch.Tensor:
    """Evaluate every fixed ping under every receiver baseline state."""
    if states.ndim != 2 or pings.ndim != 2 or states.shape[1] != pings.shape[1]:
        raise ValueError("states and pings must be matrices with the same feature width")
    magnitude = float(magnitude)
    if magnitude <= 0:
        raise ValueError("magnitude must be positive")
    n_states = states.shape[0]
    n_pings = pings.shape[0]
    expanded_states = states.repeat_interleave(n_pings, dim=0)
    expanded_pings = pings.repeat(n_states, 1)
    plus = model.decode_readout_trace(
        expanded_states + magnitude * expanded_pings
    ).association
    minus = model.decode_readout_trace(
        expanded_states - magnitude * expanded_pings
    ).association
    response = (plus - minus) / (2.0 * magnitude)
    return response.reshape(n_states, n_pings, -1)


def diagonal_advantage(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("alignment matrix must be square")
    count = matrix.shape[0]
    if count < 2:
        raise ValueError("state crossing requires at least two records")
    diagonal = torch.diagonal(matrix).mean()
    off_mask = ~torch.eye(count, dtype=torch.bool, device=matrix.device)
    off_diagonal = matrix[off_mask].mean()
    return diagonal, off_diagonal, diagonal - off_diagonal


def _alignment_matrix(responses: torch.Tensor, targets: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if responses.ndim != 3 or targets.ndim != 2:
        raise ValueError("responses must be [states,pings,features] and targets [states,features]")
    if responses.shape[0] != targets.shape[0] or responses.shape[2] != targets.shape[1]:
        raise ValueError("response and target dimensions must match")
    expanded_targets = targets[:, None, :]
    numerator = torch.sum(responses * expanded_targets, dim=2)
    response_norm = torch.linalg.vector_norm(responses, dim=2)
    target_norm = torch.linalg.vector_norm(targets, dim=1)[:, None]
    denominator = response_norm * target_norm
    return numerator / torch.clamp(denominator, min=eps)


def _project_cross(responses: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    states, pings, features = responses.shape
    flat = responses.reshape(states * pings, features)
    projected = center_rows(apply_listener(flat, weight))
    return projected.reshape(states, pings, -1)


def _matrix_tuple(matrix: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in matrix.detach().cpu().tolist()
    )


def _percentile(value: float, controls: Sequence[float]) -> float:
    less = sum(control < value for control in controls)
    equal = sum(control == value for control in controls)
    return (less + 0.5 * equal + 0.5) / (len(controls) + 1.0)


def _metric(
    real_alignment: torch.Tensor,
    linear_alignment: torch.Tensor,
    control_advantages: Sequence[float],
) -> StateCrossMetric:
    diagonal, off_diagonal, advantage = diagonal_advantage(real_alignment)
    _linear_diagonal, _linear_off, linear_advantage = diagonal_advantage(linear_alignment)
    advantage_value = float(advantage.item())
    return StateCrossMetric(
        alignment_matrix=_matrix_tuple(real_alignment),
        diagonal_mean=float(diagonal.item()),
        off_diagonal_mean=float(off_diagonal.item()),
        diagonal_advantage=advantage_value,
        control_advantage_mean=mean(control_advantages),
        control_advantage_percentile=_percentile(advantage_value, control_advantages),
        control_advantages=tuple(float(value) for value in control_advantages),
        linear_alignment_matrix=_matrix_tuple(linear_alignment),
        linear_diagonal_advantage=float(linear_advantage.item()),
    )


def evaluate_state_crossing(
    fens: Iterable[str],
    model,
    *,
    rho: float = 0.75,
    magnitudes: Sequence[float] = (0.5, 1.0, 2.0),
    controls: int = 32,
    seed: int = 0,
) -> StateCrossResult:
    """Cross step-4 history pings over receiver states and predict the held-out step-5 readout transition."""
    fen_list = tuple(fens)
    if len(fen_list) < 2:
        raise ValueError("state-crossing requires at least two positions")
    if controls < 1:
        raise ValueError("controls must be at least one")
    magnitude_values = tuple(float(value) for value in magnitudes)
    if not magnitude_values or any(value <= 0 for value in magnitude_values):
        raise ValueError("magnitudes must be positive")

    memory_states = []
    history_pings = []
    final_states = []
    for index, fen in enumerate(fen_list):
        canonical, _mirrored = canonical_fen(fen)
        features = torch.tensor(
            encode_fen(canonical), dtype=torch.float32, device=model.device
        )
        baseline = model.forward(features, include_activity=True)
        if getattr(baseline, "instability", None) is not None:
            raise RuntimeError(f"baseline instability at position {index}: {baseline.instability}")
        activity = tuple(baseline.activity)
        if len(activity) < 2:
            raise ValueError("state-crossing requires at least two settling steps")
        memory = orthogonal_history_readout(
            activity[:-1], rho=rho, readout_indices=model.readout_index
        )
        final = activity[-1].index_select(1, model.readout_index)
        memory_states.append(memory.current)
        history_pings.append(memory.orthogonal)
        final_states.append(final)

    states = torch.cat(memory_states, dim=0)
    pings = torch.cat(history_pings, dim=0)
    final = torch.cat(final_states, dim=0)
    memory_trace = model.decode_readout_trace(states)
    final_trace = model.decode_readout_trace(final)

    association_target = final_trace.association - memory_trace.association
    value_target = center_rows(final_trace.value_logits - memory_trace.value_logits)
    policy_target = center_rows(final_trace.policy_logits - memory_trace.policy_logits)

    decoder_weight = model.tensors["decoder.weight"]
    value_weight = model.tensors["value.weight"]
    policy_weight = model.tensors["policy.weight"]
    linear_ping = pings @ decoder_weight.T
    linear_association = linear_ping.unsqueeze(0).expand(len(fen_list), -1, -1)
    linear_value = _project_cross(linear_association, value_weight)
    linear_policy = _project_cross(linear_association, policy_weight)
    linear_assoc_alignment = _alignment_matrix(linear_association, association_target)
    linear_value_alignment = _alignment_matrix(linear_value, value_target)
    linear_policy_alignment = _alignment_matrix(linear_policy, policy_target)

    control_pings = tuple(
        shuffled_orthogonal_control(states, pings, seed=int(seed) + control_index + 1)
        for control_index in range(int(controls))
    )

    runs = []
    for magnitude in magnitude_values:
        association_response = cross_odd_responses(
            model, states, pings, magnitude=magnitude
        )
        value_response = _project_cross(association_response, value_weight)
        policy_response = _project_cross(association_response, policy_weight)
        assoc_alignment = _alignment_matrix(association_response, association_target)
        value_alignment = _alignment_matrix(value_response, value_target)
        policy_alignment = _alignment_matrix(policy_response, policy_target)

        assoc_control_advantages = []
        value_control_advantages = []
        policy_control_advantages = []
        for control_ping in control_pings:
            control_association = cross_odd_responses(
                model, states, control_ping, magnitude=magnitude
            )
            control_value = _project_cross(control_association, value_weight)
            control_policy = _project_cross(control_association, policy_weight)
            _d, _o, advantage = diagonal_advantage(
                _alignment_matrix(control_association, association_target)
            )
            assoc_control_advantages.append(float(advantage.item()))
            _d, _o, advantage = diagonal_advantage(
                _alignment_matrix(control_value, value_target)
            )
            value_control_advantages.append(float(advantage.item()))
            _d, _o, advantage = diagonal_advantage(
                _alignment_matrix(control_policy, policy_target)
            )
            policy_control_advantages.append(float(advantage.item()))

        runs.append(
            StateCrossRun(
                magnitude=magnitude,
                association=_metric(
                    assoc_alignment,
                    linear_assoc_alignment,
                    assoc_control_advantages,
                ),
                value=_metric(
                    value_alignment,
                    linear_value_alignment,
                    value_control_advantages,
                ),
                policy=_metric(
                    policy_alignment,
                    linear_policy_alignment,
                    policy_control_advantages,
                ),
            )
        )

    return StateCrossResult(
        rho=float(rho),
        positions=len(fen_list),
        controls=int(controls),
        seed=int(seed),
        runs=tuple(runs),
    )


__all__ = [
    "StateCrossMetric",
    "StateCrossResult",
    "StateCrossRun",
    "cross_odd_responses",
    "diagonal_advantage",
    "evaluate_state_crossing",
]
