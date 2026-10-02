from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RetrievalDetails:
    scores: tuple[tuple[float, ...], ...]
    ranks: tuple[float, ...]
    margins: tuple[float, ...]


def center_rows(values: torch.Tensor) -> torch.Tensor:
    """Remove the common-output direction from each record."""
    if values.ndim != 2:
        raise ValueError("values must be a matrix")
    return values - values.mean(dim=1, keepdim=True)


def retrieval_details(scores: torch.Tensor) -> RetrievalDetails:
    """Return the full score matrix plus tie-aware correct ranks and margins."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square matrix")
    count = scores.shape[0]
    if count < 2:
        raise ValueError("retrieval requires at least two memories")

    ranks: list[float] = []
    margins: list[float] = []
    for row in range(count):
        correct = scores[row, row]
        others = torch.cat((scores[row, :row], scores[row, row + 1 :]))
        greater = torch.sum(others > correct).float()
        equal = torch.sum(others == correct).float()
        ranks.append(float((1.0 + greater + 0.5 * equal).item()))
        margins.append(float((correct - torch.max(others)).item()))

    return RetrievalDetails(
        scores=tuple(
            tuple(float(value) for value in row)
            for row in scores.detach().cpu().tolist()
        ),
        ranks=tuple(ranks),
        margins=tuple(margins),
    )


__all__ = ["RetrievalDetails", "center_rows", "retrieval_details"]
