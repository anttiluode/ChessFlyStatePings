import numpy as np
import pytest
import torch

from chessfly_statepings.graph import ChessFlyGraph
from chessfly_statepings.model import ChessFlyBaseline, DeviceError, resolve_device
from chessfly_statepings.weights import ChessFlyWeights


def toy_graph():
    return ChessFlyGraph(
        node_count=2,
        row_ptr=np.array([0, 0, 2], dtype=np.int64),
        col_idx=np.array([0, 1], dtype=np.int64),
        sign=np.array([1, 1], dtype=np.int16),
        groups=np.array([2, 3], dtype=np.uint8),
        inputs=np.array([0], dtype=np.int64),
        readout=np.array([1], dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
    )


def toy_weights(*, nan_scale=False):
    t = {
        "encoder.weight": torch.zeros((1, 780)),
        "encoder.bias": torch.zeros(1),
        "log_gain": torch.zeros(2),
        "scale": torch.ones((3, 2)),
        "shift": torch.zeros((3, 2)),
        "decoder.weight": torch.ones((2, 1)),
        "decoder.bias": torch.zeros(2),
        "policy.weight": torch.zeros((1968, 2)),
        "policy.bias": torch.zeros(1968),
        "value.weight": torch.zeros((64, 2)),
        "value.bias": torch.zeros(64),
    }
    t["encoder.weight"][0, 0] = 1.0
    if nan_scale:
        t["scale"][1, 0] = float("nan")
    return ChessFlyWeights.from_tensors(t, {"steps": "3", "alpha": "1", "hidden": "2"})


def test_baseline_recurrence_matches_known_three_step_trajectory():
    model = ChessFlyBaseline(toy_graph(), toy_weights())
    x = torch.zeros(780); x[0] = 1
    result = model.forward(x, include_activity=True)
    assert result.instability is None
    assert len(result.activity) == 3
    assert torch.allclose(result.activity[0][0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(result.activity[1][0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(result.activity[2][0], torch.tensor([1.0, 2.0]))
    assert result.policy_logits.shape == (1, 1968)
    assert result.value_logits.shape == (1, 64)


def test_baseline_accepts_batches():
    model = ChessFlyBaseline(toy_graph(), toy_weights())
    result = model.forward(torch.zeros((4, 780)))
    assert result.policy_logits.shape == (4, 1968)


def test_resolve_device_auto_and_explicit_unavailable_cuda():
    assert str(resolve_device("auto")) in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        with pytest.raises(DeviceError, match="CUDA"):
            resolve_device("cuda")


def test_nonfinite_activity_is_reported_not_decoded():
    model = ChessFlyBaseline(toy_graph(), toy_weights(nan_scale=True))
    x = torch.zeros(780); x[0] = 1
    result = model.forward(x)
    assert result.instability is not None
    assert result.policy_logits is None
    assert result.value_logits is None
