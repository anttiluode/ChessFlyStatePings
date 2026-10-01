"""Legal-move policy wrapper around baseline or StatePing forward models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .encoding import ACTION_INDEX, canonical_fen, encode_fen, mirror_uci
from .state_ping import trajectory_summary


class PolicyError(RuntimeError):
    pass


class PolicyInstabilityError(PolicyError):
    pass


@dataclass(frozen=True, slots=True)
class PolicyCandidate:
    uci: str
    probability: float
    san: str


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    move: Any
    selected_uci: str
    top: tuple[PolicyCandidate, ...]
    legal_probabilities: Mapping[str, float]
    value_expectation: float
    activity: Mapping[str, float]


class ChessFlyPolicy:
    def __init__(self, model: Any, *, forward_kwargs: Mapping[str, Any] | None = None) -> None:
        self.model = model
        self.forward_kwargs = dict(forward_kwargs or {})

    def select(self, board: Any) -> PolicyDecision:
        moves = tuple(board.legal_moves)
        if not moves:
            raise PolicyError("cannot select from a terminal position")
        canonical, mirrored = canonical_fen(board.fen())
        canonical_moves = [mirror_uci(move.uci()) if mirrored else move.uci() for move in moves]
        try:
            indices = [ACTION_INDEX[uci] for uci in canonical_moves]
        except KeyError as exc:
            raise PolicyError(f"legal move is outside ChessFly action space: {exc.args[0]}") from exc
        features = torch.tensor(encode_fen(canonical), dtype=torch.float32)
        result = self.model.forward(features, include_activity=True, **self.forward_kwargs)
        if result.instability is not None or result.policy_logits is None or result.value_logits is None:
            raise PolicyInstabilityError(result.instability or "model produced no logits")
        logits = result.policy_logits[0]
        idx = torch.as_tensor(indices, dtype=torch.int64, device=logits.device)
        probs = torch.softmax(logits.index_select(0, idx), dim=0).detach().cpu().tolist()
        ranked = sorted(zip(moves, canonical_moves, probs), key=lambda item: (-float(item[2]), item[1]))
        selected = ranked[0][0]
        legal_probabilities = {move.uci(): float(prob) for move, _canonical, prob in ranked}
        top = tuple(PolicyCandidate(move.uci(), float(prob), board.san(move)) for move, _canonical, prob in ranked[:5])
        value_probs = torch.softmax(result.value_logits[0], dim=0).detach().cpu().tolist()
        value = sum(float(p) * ((i + 0.5) / 64.0) for i, p in enumerate(value_probs))
        activity: dict[str, float] = {}
        if result.activity:
            final = result.activity[-1]
            activity["active_fraction"] = float((final > 0).float().mean().item())
            activity["mean_activity"] = float(final.float().mean().item())
            activity["peak_activity"] = float(final.float().max().item())
            if "rho" in self.forward_kwargs:
                readout = getattr(self.model, "readout_index", None)
                summary = trajectory_summary(result.activity, rho=float(self.forward_kwargs["rho"]), readout_indices=readout)
                if summary["steps"]:
                    activity.update({f"residue_{k}": float(v) for k, v in summary["steps"][-1].items()})
        return PolicyDecision(selected, selected.uci(), top, legal_probabilities, float(value), activity)


__all__ = ["ChessFlyPolicy", "PolicyCandidate", "PolicyDecision", "PolicyError", "PolicyInstabilityError"]
