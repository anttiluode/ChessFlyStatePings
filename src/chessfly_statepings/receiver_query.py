from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping, Sequence

import torch

from .encoding import canonical_fen, encode_fen
from .orthogonal import orthogonal_history_readout, shuffled_orthogonal_control
from .query_memory import RetrievalMetrics, retrieval_metrics, score_queries


@dataclass(frozen=True, slots=True)
class ReceiverQueryRun:
    magnitude: float
    association: RetrievalMetrics
    value: RetrievalMetrics
    association_control_mean: RetrievalMetrics
    value_control_mean: RetrievalMetrics
    association_percentiles: Mapping[str, float]
    value_percentiles: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ReceiverGeometryResult:
    positions: int
    controls: int
    seed: int
    raw_history: RetrievalMetrics
    present_receiver: RetrievalMetrics
    runs: tuple[ReceiverQueryRun, ...]


@dataclass(frozen=True, slots=True)
class ReceiverQueryResult:
    rho: float
    positions: int
    controls: int
    seed: int
    raw_history: RetrievalMetrics
    present_receiver: RetrievalMetrics
    runs: tuple[ReceiverQueryRun, ...]


def receiver_signature(model, current: torch.Tensor, direction: torch.Tensor, *, magnitude: float):
    if current.shape != direction.shape:
        raise ValueError("current and direction must have the same shape")
    if current.ndim != 2:
        raise ValueError("current and direction must have shape [batch, features]")
    magnitude = float(magnitude)
    if magnitude <= 0:
        raise ValueError("magnitude must be positive")
    plus = model.decode_readout_trace(current + magnitude * direction)
    minus = model.decode_readout_trace(current - magnitude * direction)
    scale = 1.0 / (2.0 * magnitude)
    return (
        (plus.association - minus.association) * scale,
        (plus.value_logits - minus.value_logits) * scale,
    )


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


def _metric_percentiles(real: RetrievalMetrics, controls: Sequence[RetrievalMetrics]) -> dict[str, float]:
    return {
        key: _percentile(getattr(real, key), [getattr(item, key) for item in controls])
        for key in ("accuracy", "mean_reciprocal_rank", "mean_correct_margin")
    }


def evaluate_receiver_geometry(
    model,
    *,
    memory_current: torch.Tensor,
    memory_history: torch.Tensor,
    query_current: torch.Tensor,
    query_history: torch.Tensor,
    magnitudes: Sequence[float] = (1.0, 2.0, 4.0),
    controls: int = 32,
    seed: int = 0,
) -> ReceiverGeometryResult:
    if controls < 1:
        raise ValueError("controls must be at least 1")
    if not (memory_current.shape == memory_history.shape == query_current.shape == query_history.shape):
        raise ValueError("memory/query present/history tensors must share shape")
    if memory_current.ndim != 2 or memory_current.shape[0] < 2:
        raise ValueError("receiver-query requires at least two rows")
    magnitude_values = tuple(float(value) for value in magnitudes)
    if not magnitude_values or any(value <= 0 for value in magnitude_values):
        raise ValueError("magnitudes must be positive")

    raw_history = retrieval_metrics(score_queries(query_history, memory_history))
    memory_present_trace = model.decode_readout_trace(memory_current)
    query_present_trace = model.decode_readout_trace(query_current)
    present_receiver = retrieval_metrics(
        score_queries(query_present_trace.association, memory_present_trace.association)
    )

    memory_controls = [
        shuffled_orthogonal_control(memory_current, memory_history, seed=int(seed) + i + 1)
        for i in range(controls)
    ]
    query_controls = [
        shuffled_orthogonal_control(query_current, query_history, seed=int(seed) + i + 1)
        for i in range(controls)
    ]
    memory_directions = torch.cat([memory_history, *memory_controls], dim=0)
    query_directions = torch.cat([query_history, *query_controls], dim=0)
    memory_expanded = memory_current.repeat(controls + 1, 1)
    query_expanded = query_current.repeat(controls + 1, 1)
    positions = memory_current.shape[0]

    runs = []
    for magnitude in magnitude_values:
        memory_assoc_all, memory_value_all = receiver_signature(
            model, memory_expanded, memory_directions, magnitude=magnitude
        )
        query_assoc_all, query_value_all = receiver_signature(
            model, query_expanded, query_directions, magnitude=magnitude
        )
        memory_assoc_all = memory_assoc_all.reshape(controls + 1, positions, -1)
        query_assoc_all = query_assoc_all.reshape(controls + 1, positions, -1)
        memory_value_all = memory_value_all.reshape(controls + 1, positions, -1)
        query_value_all = query_value_all.reshape(controls + 1, positions, -1)

        assoc_real = retrieval_metrics(score_queries(query_assoc_all[0], memory_assoc_all[0]))
        value_real = retrieval_metrics(score_queries(query_value_all[0], memory_value_all[0]))
        assoc_controls = [
            retrieval_metrics(score_queries(query_assoc_all[i], memory_assoc_all[i]))
            for i in range(1, controls + 1)
        ]
        value_controls = [
            retrieval_metrics(score_queries(query_value_all[i], memory_value_all[i]))
            for i in range(1, controls + 1)
        ]
        runs.append(ReceiverQueryRun(
            magnitude=magnitude,
            association=assoc_real,
            value=value_real,
            association_control_mean=_mean_metrics(assoc_controls),
            value_control_mean=_mean_metrics(value_controls),
            association_percentiles=_metric_percentiles(assoc_real, assoc_controls),
            value_percentiles=_metric_percentiles(value_real, value_controls),
        ))
    return ReceiverGeometryResult(
        positions=memory_current.shape[0],
        controls=int(controls),
        seed=int(seed),
        raw_history=raw_history,
        present_receiver=present_receiver,
        runs=tuple(runs),
    )


def evaluate_receiver_query(
    fens: Iterable[str],
    model,
    *,
    rho: float = 0.75,
    magnitudes: Sequence[float] = (1.0, 2.0, 4.0),
    controls: int = 32,
    seed: int = 0,
) -> ReceiverQueryResult:
    fen_list = tuple(fens)
    if len(fen_list) < 2:
        raise ValueError("receiver-query requires at least two positions")

    memory_current = []
    memory_history = []
    query_current = []
    query_history = []
    for position_index, fen in enumerate(fen_list):
        canonical, _mirrored = canonical_fen(fen)
        features = torch.tensor(encode_fen(canonical), dtype=torch.float32, device=model.device)
        baseline = model.forward(features, include_activity=True)
        if getattr(baseline, "instability", None) is not None:
            raise RuntimeError(f"baseline instability at position {position_index}: {baseline.instability}")
        activity = tuple(baseline.activity)
        if len(activity) < 2:
            raise ValueError("receiver-query requires at least two settling steps")
        memory = orthogonal_history_readout(activity[:-1], rho=rho, readout_indices=model.readout_index)
        query = orthogonal_history_readout(activity, rho=rho, readout_indices=model.readout_index)
        memory_current.append(memory.current)
        memory_history.append(memory.orthogonal)
        query_current.append(query.current)
        query_history.append(query.orthogonal)

    geometry = evaluate_receiver_geometry(
        model,
        memory_current=torch.cat(memory_current, dim=0),
        memory_history=torch.cat(memory_history, dim=0),
        query_current=torch.cat(query_current, dim=0),
        query_history=torch.cat(query_history, dim=0),
        magnitudes=magnitudes,
        controls=controls,
        seed=seed,
    )
    return ReceiverQueryResult(
        rho=float(rho),
        positions=geometry.positions,
        controls=geometry.controls,
        seed=geometry.seed,
        raw_history=geometry.raw_history,
        present_receiver=geometry.present_receiver,
        runs=geometry.runs,
    )


__all__ = [
    "ReceiverGeometryResult",
    "ReceiverQueryResult",
    "ReceiverQueryRun",
    "evaluate_receiver_geometry",
    "evaluate_receiver_query",
    "receiver_signature",
]
