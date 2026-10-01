"""Fast/slow trajectory probe and StatePing recurrence."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .graph import ChessFlyGraph
from .model import ChessFlyBaseline, ForwardResult
from .weights import ChessFlyWeights


def _stats(hidden: torch.Tensor, residue: torch.Tensor) -> dict[str, float]:
    h = hidden.float()
    r = residue.float()
    mean_abs = float(r.abs().mean().item())
    rms = float(torch.sqrt(torch.mean(r * r)).item())
    active = float((h > 0).float().mean().item())
    hflat, rflat = h.reshape(-1), r.reshape(-1)
    denom = float((torch.linalg.vector_norm(hflat) * torch.linalg.vector_norm(rflat)).item())
    cosine = 0.0 if denom == 0.0 else float(torch.dot(hflat, rflat).item() / denom)
    return {"mean_abs_residue": mean_abs, "rms_residue": rms, "active_fraction": active, "h_residue_cosine": cosine}


def trajectory_summary(activity: Sequence[torch.Tensor], *, rho: float, readout_indices: torch.Tensor | Sequence[int] | None = None) -> dict:
    if not 0 <= rho < 1:
        raise ValueError("rho must satisfy 0 <= rho < 1")
    if not activity:
        return {"rho": float(rho), "steps": []}
    slow = torch.zeros_like(activity[0])
    if readout_indices is not None:
        idx = torch.as_tensor(readout_indices, dtype=torch.int64, device=activity[0].device)
    else:
        idx = None
    steps = []
    for h in activity:
        slow = rho * slow + (1.0 - rho) * h
        residue = h - slow
        entry = _stats(h, residue)
        if idx is not None and idx.numel() > 0:
            h_r = h.index_select(1, idx)
            r_r = residue.index_select(1, idx)
            for key, value in _stats(h_r, r_r).items():
                entry[f"readout_{key}"] = value
        steps.append(entry)
    return {"rho": float(rho), "steps": steps}


class StatePingModel(ChessFlyBaseline):
    """Frozen ChessFly with a global fast-minus-slow recurrent message."""

    def __init__(self, graph: ChessFlyGraph, weights: ChessFlyWeights, *, device: str | torch.device = "auto") -> None:
        super().__init__(graph, weights, device=device)

    def forward(self, features: Any, *, rho: float = 0.75, kappa: float = 0.0, include_activity: bool = False) -> ForwardResult:
        if not 0 <= rho < 1:
            raise ValueError("rho must satisfy 0 <= rho < 1")
        if not math.isfinite(float(kappa)):
            raise ValueError("kappa must be finite")
        with torch.inference_mode():
            _, drive = self._prepare(features)
            hidden = torch.zeros_like(drive)
            slow = torch.zeros_like(drive)
            residue = torch.zeros_like(drive)
            activity: list[torch.Tensor] = []
            for step in range(self.weights.steps):
                recurrent = torch.sparse.mm(self.matrix, hidden.transpose(0, 1)).transpose(0, 1)
                if kappa == 0.0:
                    mixed = recurrent
                else:
                    residue_recurrent = torch.sparse.mm(self.matrix, residue.transpose(0, 1)).transpose(0, 1)
                    mixed = recurrent + float(kappa) * residue_recurrent
                pre = (mixed + drive) * self.tensors["scale"][step] + self.tensors["shift"][step]
                hidden = (1.0 - self.weights.alpha) * hidden + self.weights.alpha * torch.relu(pre)
                if not torch.isfinite(hidden).all():
                    return ForwardResult(None, None, tuple(activity), f"non-finite activity at step {step + 1}")
                slow = float(rho) * slow + (1.0 - float(rho)) * hidden
                residue = hidden - slow
                if include_activity:
                    activity.append(hidden.clone())
            return self._decode(hidden, activity)


__all__ = ["StatePingModel", "trajectory_summary"]
