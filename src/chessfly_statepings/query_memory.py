from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .encoding import canonical_fen, encode_fen
from .orthogonal import orthogonal_history_readout


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    accuracy: float
    mean_reciprocal_rank: float
    mean_correct_margin: float


@dataclass(frozen=True, slots=True)
class QueryMemoryResult:
    rho: float
    seed: int
    positions: int
    present: RetrievalMetrics
    history: RetrievalMetrics
    shuffled_history: RetrievalMetrics


def _normalize_rows(values: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values must have shape [items, features]")
    values = values.float()
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    return values / torch.clamp(norms, min=eps)


def score_queries(
    present_queries: torch.Tensor,
    present_keys: torch.Tensor,
    history_queries: torch.Tensor | None = None,
    history_keys: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cosine retrieval scores, optionally adding an equal-weight history coordinate."""
    if present_queries.ndim != 2 or present_keys.ndim != 2:
        raise ValueError("present queries/keys must be matrices")
    if present_queries.shape[1] != present_keys.shape[1]:
        raise ValueError("present query/key feature dimensions must match")
    scores = _normalize_rows(present_queries) @ _normalize_rows(present_keys).T
    if (history_queries is None) != (history_keys is None):
        raise ValueError("history queries and keys must be provided together")
    if history_queries is not None and history_keys is not None:
        if history_queries.ndim != 2 or history_keys.ndim != 2:
            raise ValueError("history queries/keys must be matrices")
        if history_queries.shape[0] != present_queries.shape[0] or history_keys.shape[0] != present_keys.shape[0]:
            raise ValueError("history and present item counts must match")
        if history_queries.shape[1] != history_keys.shape[1]:
            raise ValueError("history query/key feature dimensions must match")
        history_scores = _normalize_rows(history_queries) @ _normalize_rows(history_keys).T
        scores = 0.5 * (scores + history_scores)
    return scores


def retrieval_metrics(scores: torch.Tensor) -> RetrievalMetrics:
    """Assume query i should retrieve memory i; report top-1, MRR, and correct margin."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square [queries, memories] matrix")
    count = scores.shape[0]
    if count < 2:
        raise ValueError("retrieval requires at least two memories")
    diagonal = torch.diagonal(scores)
    predictions = torch.argmax(scores, dim=1)
    expected = torch.arange(count, device=scores.device)
    accuracy = float((predictions == expected).float().mean().item())

    reciprocal_ranks = []
    margins = []
    for row in range(count):
        correct = diagonal[row]
        others = torch.cat((scores[row, :row], scores[row, row + 1 :]))
        greater = torch.sum(others > correct).float()
        equal = torch.sum(others == correct).float()
        rank = 1.0 + greater + 0.5 * equal
        reciprocal_ranks.append(1.0 / rank)
        margins.append(correct - torch.max(others))
    return RetrievalMetrics(
        accuracy=accuracy,
        mean_reciprocal_rank=float(torch.stack(reciprocal_ranks).mean().item()),
        mean_correct_margin=float(torch.stack(margins).mean().item()),
    )


def _shuffled_rows(values: torch.Tensor, *, seed: int) -> torch.Tensor:
    count = values.shape[0]
    if count < 2:
        raise ValueError("query-memory requires at least two positions")
    shift = 1 + (int(seed) % (count - 1))
    return values.roll(shifts=shift, dims=0)


def evaluate_query_memory(
    fens: Iterable[str],
    model,
    *,
    rho: float = 0.75,
    seed: int = 0,
) -> QueryMemoryResult:
    """Ask whether the final state-bearing ping retrieves its own step-(K-1) history.

    The memory bank stores each position's penultimate settling state plus its
    orthogonal fast-minus-slow history coordinate. The final settling state is
    used as the query. No model weights are changed and no retrieval parameters
    are trained.
    """
    fen_list = tuple(fens)
    if len(fen_list) < 2:
        raise ValueError("query-memory requires at least two positions")

    present_keys = []
    history_keys = []
    present_queries = []
    history_queries = []

    for index, fen in enumerate(fen_list):
        canonical, _mirrored = canonical_fen(fen)
        features = torch.tensor(encode_fen(canonical), dtype=torch.float32, device=model.device)
        baseline = model.forward(features, include_activity=True)
        if getattr(baseline, "instability", None) is not None:
            raise RuntimeError(f"baseline instability at position {index}: {baseline.instability}")
        activity = tuple(baseline.activity)
        if len(activity) < 2:
            raise ValueError("query-memory requires at least two settling steps")

        memory = orthogonal_history_readout(
            activity[:-1], rho=rho, readout_indices=model.readout_index
        )
        query = orthogonal_history_readout(
            activity, rho=rho, readout_indices=model.readout_index
        )
        present_keys.append(memory.current)
        history_keys.append(memory.orthogonal)
        present_queries.append(query.current)
        history_queries.append(query.orthogonal)

    present_key = torch.cat(present_keys, dim=0)
    history_key = torch.cat(history_keys, dim=0)
    present_query = torch.cat(present_queries, dim=0)
    history_query = torch.cat(history_queries, dim=0)

    present_scores = score_queries(present_query, present_key)
    history_scores = score_queries(present_query, present_key, history_query, history_key)
    shuffled_scores = score_queries(
        present_query,
        present_key,
        _shuffled_rows(history_query, seed=seed),
        history_key,
    )

    return QueryMemoryResult(
        rho=float(rho),
        seed=int(seed),
        positions=len(fen_list),
        present=retrieval_metrics(present_scores),
        history=retrieval_metrics(history_scores),
        shuffled_history=retrieval_metrics(shuffled_scores),
    )


__all__ = [
    "QueryMemoryResult",
    "RetrievalMetrics",
    "evaluate_query_memory",
    "retrieval_metrics",
    "score_queries",
]
