from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping, Sequence

import torch

from .encoding import canonical_fen, encode_fen
from .orthogonal import orthogonal_history_readout
from .query_memory import RetrievalMetrics, retrieval_metrics, score_queries
from .receiver_query import receiver_signature


@dataclass(frozen=True, slots=True)
class ListenerSignatureResult:
    association: RetrievalMetrics
    value: RetrievalMetrics
    policy: RetrievalMetrics
    value_control_mean: RetrievalMetrics
    policy_control_mean: RetrievalMetrics
    value_percentiles: Mapping[str, float]
    policy_percentiles: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ListenerReplacementRun:
    magnitude: float
    association: RetrievalMetrics
    value: RetrievalMetrics
    policy: RetrievalMetrics
    value_control_mean: RetrievalMetrics
    policy_control_mean: RetrievalMetrics
    value_percentiles: Mapping[str, float]
    policy_percentiles: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ListenerReplacementResult:
    rho: float
    positions: int
    controls: int
    seed: int
    raw_history: RetrievalMetrics
    runs: tuple[ListenerReplacementRun, ...]


def apply_listener(signatures: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply a bias-free listener map; central differences cancel head biases."""
    if signatures.ndim != 2 or weight.ndim != 2:
        raise ValueError("signatures and weight must be matrices")
    if signatures.shape[1] != weight.shape[1]:
        raise ValueError("listener input dimension must match signature dimension")
    return signatures @ weight.T


def _listener_permutations(width: int, *, controls: int, seed: int = 0) -> tuple[torch.Tensor, ...]:
    if controls < 1:
        raise ValueError("controls must be at least 1")
    if width < 2:
        raise ValueError("listener must have at least two input coordinates")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    identity = torch.arange(width, device="cpu")
    permutations = []
    for _ in range(int(controls)):
        permutation = torch.randperm(width, generator=generator, device="cpu")
        if torch.equal(permutation, identity):
            permutation = permutation.roll(1)
        permutations.append(permutation)
    return tuple(permutations)


def shuffled_listener_controls(
    weight: torch.Tensor,
    *,
    controls: int,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Column-permute a listener, preserving its singular values exactly."""
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    return tuple(
        weight.index_select(1, permutation.to(weight.device))
        for permutation in _listener_permutations(weight.shape[1], controls=controls, seed=seed)
    )


def listener_retrieval(
    memory_signatures: torch.Tensor,
    query_signatures: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[RetrievalMetrics, RetrievalMetrics]:
    """Compare raw signature retrieval with retrieval after one listener map."""
    if memory_signatures.shape != query_signatures.shape:
        raise ValueError("memory/query signatures must share shape")
    raw = retrieval_metrics(score_queries(query_signatures, memory_signatures))
    heard = retrieval_metrics(
        score_queries(
            apply_listener(query_signatures, weight),
            apply_listener(memory_signatures, weight),
        )
    )
    return raw, heard


def _mean_metrics(items: Sequence[RetrievalMetrics]) -> RetrievalMetrics:
    return RetrievalMetrics(
        accuracy=mean(item.accuracy for item in items),
        mean_reciprocal_rank=mean(item.mean_reciprocal_rank for item in items),
        mean_correct_margin=mean(item.mean_correct_margin for item in items),
    )


def _percentile(value: float, controls: Sequence[float]) -> float:
    less = sum(float(control) < float(value) for control in controls)
    equal = sum(float(control) == float(value) for control in controls)
    return (less + 0.5 * equal + 0.5) / (len(controls) + 1.0)


def _metric_percentiles(
    real: RetrievalMetrics,
    controls: Sequence[RetrievalMetrics],
) -> dict[str, float]:
    keys = ("accuracy", "mean_reciprocal_rank", "mean_correct_margin")
    return {
        key: _percentile(getattr(real, key), [getattr(item, key) for item in controls])
        for key in keys
    }


def evaluate_listener_signatures(
    memory_signatures: torch.Tensor,
    query_signatures: torch.Tensor,
    *,
    value_weight: torch.Tensor,
    policy_weight: torch.Tensor,
    controls: int = 32,
    seed: int = 0,
) -> ListenerSignatureResult:
    """Hold the temporal signature fixed and replace only the downstream listener."""
    if memory_signatures.shape != query_signatures.shape:
        raise ValueError("memory/query signatures must share shape")
    if memory_signatures.ndim != 2 or memory_signatures.shape[0] < 2:
        raise ValueError("listener replacement requires at least two records")

    association = retrieval_metrics(score_queries(query_signatures, memory_signatures))
    value = retrieval_metrics(
        score_queries(
            apply_listener(query_signatures, value_weight),
            apply_listener(memory_signatures, value_weight),
        )
    )
    policy = retrieval_metrics(
        score_queries(
            apply_listener(query_signatures, policy_weight),
            apply_listener(memory_signatures, policy_weight),
        )
    )

    value_controls = []
    for permutation in _listener_permutations(
        value_weight.shape[1], controls=controls, seed=seed
    ):
        control = value_weight.index_select(1, permutation.to(value_weight.device))
        value_controls.append(
            retrieval_metrics(
                score_queries(
                    apply_listener(query_signatures, control),
                    apply_listener(memory_signatures, control),
                )
            )
        )

    policy_controls = []
    for permutation in _listener_permutations(
        policy_weight.shape[1], controls=controls, seed=seed + 1_000_003
    ):
        control = policy_weight.index_select(1, permutation.to(policy_weight.device))
        policy_controls.append(
            retrieval_metrics(
                score_queries(
                    apply_listener(query_signatures, control),
                    apply_listener(memory_signatures, control),
                )
            )
        )

    return ListenerSignatureResult(
        association=association,
        value=value,
        policy=policy,
        value_control_mean=_mean_metrics(value_controls),
        policy_control_mean=_mean_metrics(policy_controls),
        value_percentiles=_metric_percentiles(value, value_controls),
        policy_percentiles=_metric_percentiles(policy, policy_controls),
    )


def evaluate_listener_replacement(
    fens: Iterable[str],
    model,
    *,
    rho: float = 0.75,
    magnitudes: Sequence[float] = (1.0, 2.0, 4.0),
    controls: int = 32,
    seed: int = 0,
) -> ListenerReplacementResult:
    """Ask whether the same real temporal signature means different things to different frozen heads."""
    fen_list = tuple(fens)
    if len(fen_list) < 2:
        raise ValueError("listener-replacement requires at least two positions")
    magnitude_values = tuple(float(value) for value in magnitudes)
    if not magnitude_values or any(value <= 0 for value in magnitude_values):
        raise ValueError("magnitudes must be positive")

    memory_current = []
    memory_history = []
    query_current = []
    query_history = []
    for index, fen in enumerate(fen_list):
        canonical, _mirrored = canonical_fen(fen)
        features = torch.tensor(
            encode_fen(canonical), dtype=torch.float32, device=model.device
        )
        baseline = model.forward(features, include_activity=True)
        if getattr(baseline, "instability", None) is not None:
            raise RuntimeError(
                f"baseline instability at position {index}: {baseline.instability}"
            )
        activity = tuple(baseline.activity)
        if len(activity) < 2:
            raise ValueError("listener-replacement requires at least two settling steps")
        memory = orthogonal_history_readout(
            activity[:-1], rho=rho, readout_indices=model.readout_index
        )
        query = orthogonal_history_readout(
            activity, rho=rho, readout_indices=model.readout_index
        )
        memory_current.append(memory.current)
        memory_history.append(memory.orthogonal)
        query_current.append(query.current)
        query_history.append(query.orthogonal)

    memory_current_tensor = torch.cat(memory_current, dim=0)
    memory_history_tensor = torch.cat(memory_history, dim=0)
    query_current_tensor = torch.cat(query_current, dim=0)
    query_history_tensor = torch.cat(query_history, dim=0)
    raw_history = retrieval_metrics(
        score_queries(query_history_tensor, memory_history_tensor)
    )

    value_weight = model.tensors["value.weight"]
    policy_weight = model.tensors["policy.weight"]
    runs = []
    for magnitude in magnitude_values:
        memory_association, _memory_value = receiver_signature(
            model,
            memory_current_tensor,
            memory_history_tensor,
            magnitude=magnitude,
        )
        query_association, _query_value = receiver_signature(
            model,
            query_current_tensor,
            query_history_tensor,
            magnitude=magnitude,
        )
        signature_result = evaluate_listener_signatures(
            memory_association,
            query_association,
            value_weight=value_weight,
            policy_weight=policy_weight,
            controls=controls,
            seed=seed,
        )
        runs.append(
            ListenerReplacementRun(
                magnitude=magnitude,
                association=signature_result.association,
                value=signature_result.value,
                policy=signature_result.policy,
                value_control_mean=signature_result.value_control_mean,
                policy_control_mean=signature_result.policy_control_mean,
                value_percentiles=signature_result.value_percentiles,
                policy_percentiles=signature_result.policy_percentiles,
            )
        )

    return ListenerReplacementResult(
        rho=float(rho),
        positions=len(fen_list),
        controls=int(controls),
        seed=int(seed),
        raw_history=raw_history,
        runs=tuple(runs),
    )


__all__ = [
    "ListenerReplacementResult",
    "ListenerReplacementRun",
    "ListenerSignatureResult",
    "apply_listener",
    "evaluate_listener_replacement",
    "evaluate_listener_signatures",
    "listener_retrieval",
    "shuffled_listener_controls",
]
