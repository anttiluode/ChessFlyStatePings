"""ChessFly safetensors checkpoint validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file

REQUIRED = frozenset({
    "decoder.bias", "decoder.weight", "encoder.bias", "encoder.weight",
    "log_gain", "policy.bias", "policy.weight", "scale", "shift",
    "value.bias", "value.weight",
})


class WeightFormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ChessFlyWeights:
    tensors: Mapping[str, torch.Tensor]
    metadata: Mapping[str, str]
    steps: int
    alpha: float
    hidden: int
    step: int | None

    @property
    def encoder_inputs(self) -> int:
        return int(self.tensors["encoder.weight"].shape[0])

    @property
    def readout_neurons(self) -> int:
        return int(self.tensors["decoder.weight"].shape[1])

    @classmethod
    def from_file(cls, path: str | Path, *, device: str = "cpu") -> "ChessFlyWeights":
        path = str(path)
        try:
            tensors = load_file(path, device=device)
            with safe_open(path, framework="pt", device=device) as handle:
                metadata = handle.metadata() or {}
        except Exception as exc:
            raise WeightFormatError(f"could not load checkpoint {path}: {exc}") from exc
        return cls.from_tensors(tensors, metadata)

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str] | None = None) -> "ChessFlyWeights":
        tensors = dict(tensors)
        metadata = dict(metadata or {})
        missing = REQUIRED - set(tensors)
        if missing:
            raise WeightFormatError("missing tensors: " + ", ".join(sorted(missing)))
        try:
            steps = int(metadata.get("steps", tensors["scale"].shape[0]))
            alpha = float(metadata.get("alpha", "0.5"))
            hidden = int(metadata.get("hidden", tensors["decoder.weight"].shape[0]))
            step = int(metadata["step"]) if "step" in metadata else None
        except (TypeError, ValueError, KeyError) as exc:
            raise WeightFormatError("invalid checkpoint metadata") from exc
        if steps < 1 or hidden < 1 or not 0 < alpha <= 1:
            raise WeightFormatError("invalid steps, hidden, or alpha")
        shapes = {name: tuple(value.shape) for name, value in tensors.items()}
        if len(shapes["encoder.weight"]) != 2 or shapes["encoder.weight"][1] != 780 or shapes["encoder.bias"] != (shapes["encoder.weight"][0],):
            raise WeightFormatError("encoder.weight/encoder.bias shape mismatch")
        if shapes["scale"] != shapes["shift"] or len(shapes["scale"]) != 2 or shapes["scale"][0] != steps:
            raise WeightFormatError("scale/shift shape mismatch")
        if shapes["decoder.weight"][0] != hidden or shapes["decoder.bias"] != (hidden,):
            raise WeightFormatError("decoder shape mismatch")
        if shapes["policy.weight"] != (1968, hidden) or shapes["policy.bias"] != (1968,):
            raise WeightFormatError("policy.weight/policy.bias shape mismatch")
        if shapes["value.weight"] != (64, hidden) or shapes["value.bias"] != (64,):
            raise WeightFormatError("value.weight/value.bias shape mismatch")
        if len(shapes["log_gain"]) != 1:
            raise WeightFormatError("log_gain must be one-dimensional")
        return cls(tensors, metadata, steps, alpha, hidden, step)


__all__ = ["ChessFlyWeights", "WeightFormatError"]
