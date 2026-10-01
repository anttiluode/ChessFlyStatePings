"""Frozen ChessFly baseline recurrence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .graph import ChessFlyGraph
from .weights import ChessFlyWeights


class DeviceError(ValueError):
    pass


class ModelShapeError(ValueError):
    pass


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        chosen = device
    elif device == "auto":
        chosen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        chosen = torch.device(device)
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise DeviceError("CUDA was requested but is not available")
    return chosen


@dataclass(frozen=True, slots=True)
class ReadoutTrace:
    association: torch.Tensor
    policy_logits: torch.Tensor
    value_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class ForwardResult:
    policy_logits: torch.Tensor | None
    value_logits: torch.Tensor | None
    activity: tuple[torch.Tensor, ...] = ()
    instability: str | None = None


class ChessFlyBaseline:
    """Exact frozen five-step-style ChessFly recurrence from a loaded checkpoint."""

    def __init__(self, graph: ChessFlyGraph, weights: ChessFlyWeights, *, device: str | torch.device = "auto") -> None:
        self.graph = graph
        self.weights = weights
        self.device = resolve_device(device)
        scale_shape = tuple(weights.tensors["scale"].shape)
        if len(scale_shape) != 2 or scale_shape[1] != graph.node_count:
            raise ModelShapeError("graph node count disagrees with scale tensor")
        if weights.encoder_inputs != len(graph.inputs):
            raise ModelShapeError("graph input count disagrees with encoder")
        if weights.readout_neurons != len(graph.readout):
            raise ModelShapeError("graph readout count disagrees with decoder")
        if tuple(weights.tensors["log_gain"].shape) != (graph.edge_count,):
            raise ModelShapeError("graph edge count disagrees with log_gain")
        self.tensors = {k: v.to(device=self.device, dtype=torch.float32) for k, v in weights.tensors.items()}
        self.input_index = torch.as_tensor(graph.inputs, dtype=torch.int64, device=self.device)
        self.readout_index = torch.as_tensor(graph.readout, dtype=torch.int64, device=self.device)
        self.matrix = graph.sparse_matrix(self.tensors["log_gain"], device=self.device)

    def _prepare(self, features: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != 780:
            raise ModelShapeError("features must have shape [batch, 780]")
        encoded = F.linear(x, self.tensors["encoder.weight"], self.tensors["encoder.bias"])
        drive = torch.zeros((x.shape[0], self.graph.node_count), dtype=torch.float32, device=self.device)
        drive.index_copy_(1, self.input_index, encoded)
        return x, drive

    def _readout_tensor(self, readout: Any) -> torch.Tensor:
        readout_tensor = torch.as_tensor(readout, dtype=torch.float32, device=self.device)
        if readout_tensor.ndim == 1:
            readout_tensor = readout_tensor.unsqueeze(0)
        if readout_tensor.ndim != 2 or readout_tensor.shape[1] != len(self.graph.readout):
            raise ModelShapeError("readout must have shape [batch, readout_neurons]")
        return readout_tensor

    def decode_readout_trace(self, readout: Any) -> ReadoutTrace:
        """Run the frozen decoder and expose its association state and both heads."""
        readout_tensor = self._readout_tensor(readout)
        association = F.gelu(
            F.linear(readout_tensor, self.tensors["decoder.weight"], self.tensors["decoder.bias"]),
            approximate="tanh",
        )
        return ReadoutTrace(
            association=association,
            policy_logits=F.linear(association, self.tensors["policy.weight"], self.tensors["policy.bias"]),
            value_logits=F.linear(association, self.tensors["value.weight"], self.tensors["value.bias"]),
        )

    def decode_readout(
        self,
        readout: Any,
        *,
        activity: Sequence[torch.Tensor] = (),
    ) -> ForwardResult:
        """Run the frozen decoder and heads from an explicit readout-neuron state."""
        trace = self.decode_readout_trace(readout)
        return ForwardResult(
            policy_logits=trace.policy_logits,
            value_logits=trace.value_logits,
            activity=tuple(activity),
            instability=None,
        )

    def _decode(self, hidden: torch.Tensor, activity: list[torch.Tensor]) -> ForwardResult:
        readout = hidden.index_select(1, self.readout_index)
        return self.decode_readout(readout, activity=activity)

    def forward(self, features: Any, *, include_activity: bool = False) -> ForwardResult:
        with torch.inference_mode():
            _, drive = self._prepare(features)
            hidden = torch.zeros_like(drive)
            activity: list[torch.Tensor] = []
            for step in range(self.weights.steps):
                recurrent = torch.sparse.mm(self.matrix, hidden.transpose(0, 1)).transpose(0, 1)
                pre = (recurrent + drive) * self.tensors["scale"][step] + self.tensors["shift"][step]
                hidden = (1.0 - self.weights.alpha) * hidden + self.weights.alpha * torch.relu(pre)
                if not torch.isfinite(hidden).all():
                    return ForwardResult(None, None, tuple(activity), f"non-finite activity at step {step + 1}")
                if include_activity:
                    activity.append(hidden.clone())
            return self._decode(hidden, activity)


__all__ = ["ChessFlyBaseline", "DeviceError", "ForwardResult", "ModelShapeError", "ReadoutTrace", "resolve_device"]
