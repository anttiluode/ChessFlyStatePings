import struct

import numpy as np
import pytest

from chessfly_statepings.graph import ChessFlyGraph, GraphFormatError


def graph_bytes(node_count=3, targets=(1, 2, 2), signs=(1, -1, 1), row_ptr=(0, 2, 3, 3)):
    header = b"CF01" + struct.pack("<II", node_count, len(targets))
    return header + np.asarray(row_ptr, dtype="<u4").tobytes() + np.asarray(targets, dtype="<u4").tobytes() + np.asarray(signs, dtype="<i2").tobytes()


def neuron_bytes(groups=(2, 1, 3)):
    positions = np.arange(len(groups) * 3, dtype="<f4")
    return positions.tobytes() + np.asarray(groups, dtype="u1").tobytes()


def test_graph_transposes_source_csr_to_target_csr_and_extracts_groups():
    g = ChessFlyGraph.from_bytes(graph_bytes(), neuron_bytes())
    assert g.node_count == 3
    assert g.edge_count == 3
    assert g.row_ptr.tolist() == [0, 0, 1, 3]
    assert g.col_idx.tolist() == [0, 0, 1]
    assert g.sign.tolist() == [1, -1, 1]
    assert g.inputs.tolist() == [0]
    assert g.readout.tolist() == [2]


def test_graph_rejects_out_of_range_target():
    with pytest.raises(GraphFormatError, match="out-of-range target"):
        ChessFlyGraph.from_bytes(graph_bytes(targets=(1, 9, 2)), neuron_bytes())


def test_graph_rejects_unknown_group():
    with pytest.raises(GraphFormatError, match="unknown group"):
        ChessFlyGraph.from_bytes(graph_bytes(), neuron_bytes((2, 7, 3)))


def test_graph_rejects_malformed_size():
    with pytest.raises(GraphFormatError, match="size mismatch"):
        ChessFlyGraph.from_bytes(graph_bytes()[:-1], neuron_bytes())
