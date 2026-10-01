import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("CHESSFLY_RUN_INTEGRATION") != "1",
    reason="set CHESSFLY_RUN_INTEGRATION=1 for real upstream assets",
)


def test_real_artifacts_load_and_baseline_is_deterministic():
    from chessfly_statepings.assets import ensure_artifacts
    from chessfly_statepings.encoding import encode_fen
    from chessfly_statepings.graph import ChessFlyGraph
    from chessfly_statepings.model import ChessFlyBaseline
    from chessfly_statepings.weights import ChessFlyWeights

    manifest = ensure_artifacts()
    graph = ChessFlyGraph.from_files(manifest.paths.connectome, manifest.paths.neurons)
    weights = ChessFlyWeights.from_file(manifest.paths.weights)
    model = ChessFlyBaseline(graph, weights, device="cpu")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    x = torch.tensor(encode_fen(fen))
    a = model.forward(x)
    b = model.forward(x)
    assert a.instability is None
    assert graph.node_count == weights.tensors["scale"].shape[1]
    assert torch.equal(a.policy_logits, b.policy_logits)
    assert torch.equal(a.value_logits, b.value_logits)
