"""Parser for the packed ChessFly graph and neuron metadata."""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


class GraphFormatError(ValueError):
    pass


def _read(path: str | Path) -> bytes:
    path = Path(path)
    data = path.read_bytes()
    if path.name.endswith(".gz"):
        try:
            return gzip.decompress(data)
        except (OSError, EOFError) as exc:
            raise GraphFormatError(f"could not decompress {path}") from exc
    return data


@dataclass(frozen=True, slots=True)
class ChessFlyGraph:
    node_count: int
    row_ptr: np.ndarray
    col_idx: np.ndarray
    sign: np.ndarray
    groups: np.ndarray
    inputs: np.ndarray
    readout: np.ndarray
    positions: np.ndarray

    @property
    def edge_count(self) -> int:
        return int(len(self.col_idx))

    @classmethod
    def from_files(cls, connectome_path: str | Path, neurons_path: str | Path) -> "ChessFlyGraph":
        return cls.from_bytes(_read(connectome_path), _read(neurons_path))

    @classmethod
    def from_bytes(cls, connectome: bytes, neurons: bytes) -> "ChessFlyGraph":
        if len(connectome) < 12:
            raise GraphFormatError("connectome shorter than header")
        node_count, edge_count = struct.unpack_from("<II", connectome, 4)
        expected = 12 + 4 * (node_count + 1) + 4 * edge_count + 2 * edge_count
        if len(connectome) != expected:
            raise GraphFormatError(f"connectome size mismatch: expected {expected}, got {len(connectome)}")
        row_offset = 12
        target_offset = row_offset + 4 * (node_count + 1)
        sign_offset = target_offset + 4 * edge_count
        source_row_ptr = np.frombuffer(connectome, dtype="<u4", count=node_count + 1, offset=row_offset).copy()
        targets = np.frombuffer(connectome, dtype="<u4", count=edge_count, offset=target_offset).copy()
        signs = np.frombuffer(connectome, dtype="<i2", count=edge_count, offset=sign_offset).copy()
        if int(source_row_ptr[0]) != 0 or int(source_row_ptr[-1]) != edge_count or np.any(source_row_ptr[1:] < source_row_ptr[:-1]):
            raise GraphFormatError("invalid source CSR row pointer")
        if np.any(targets >= node_count):
            raise GraphFormatError("connectome contains out-of-range target")
        sources = np.repeat(np.arange(node_count, dtype=np.int64), np.diff(source_row_ptr).astype(np.int64))
        order = np.argsort(targets, kind="stable")
        sorted_targets = targets[order]
        col_idx = sources[order].astype(np.int64, copy=False)
        sign = signs[order].astype(np.int16, copy=False)
        counts = np.bincount(sorted_targets.astype(np.int64), minlength=node_count)
        row_ptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64))).astype(np.int64)

        expected_neurons = 13 * node_count
        if len(neurons) != expected_neurons:
            raise GraphFormatError(f"neuron metadata size mismatch: expected {expected_neurons}, got {len(neurons)}")
        positions = np.frombuffer(neurons, dtype="<f4", count=node_count * 3, offset=0).reshape(node_count, 3).copy()
        groups = np.frombuffer(neurons, dtype="u1", count=node_count, offset=12 * node_count).copy()
        if np.any(~np.isin(groups, np.array([0,1,2,3], dtype=np.uint8))):
            raise GraphFormatError("neuron metadata contains unknown group")
        inputs = np.flatnonzero(groups == 2).astype(np.int64)
        readout = np.flatnonzero((groups != 1) & (groups != 2)).astype(np.int64)
        if len(inputs) == 0 or len(readout) == 0:
            raise GraphFormatError("graph must expose input and readout neurons")
        return cls(node_count, row_ptr, col_idx, sign, groups, inputs, readout, positions)

    def sparse_matrix(self, log_gain: torch.Tensor, *, device: str | torch.device = "cpu") -> torch.Tensor:
        if tuple(log_gain.shape) != (self.edge_count,):
            raise GraphFormatError(f"log_gain must have shape [{self.edge_count}]")
        row_ptr = torch.as_tensor(self.row_ptr, dtype=torch.int64, device=device)
        col_idx = torch.as_tensor(self.col_idx, dtype=torch.int64, device=device)
        sign = torch.as_tensor(self.sign, dtype=torch.float32, device=device)
        sign = torch.where(sign < 0, -torch.ones_like(sign), torch.ones_like(sign))
        values = sign * torch.exp(log_gain.to(device=device, dtype=torch.float32))
        return torch.sparse_csr_tensor(row_ptr, col_idx, values, size=(self.node_count, self.node_count), device=device)


__all__ = ["ChessFlyGraph", "GraphFormatError"]
