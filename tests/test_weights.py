import torch
import pytest
from safetensors.torch import save_file

from chessfly_statepings.weights import ChessFlyWeights, WeightFormatError


def tensors():
    return {
        "encoder.weight": torch.zeros((1, 780)),
        "encoder.bias": torch.zeros(1),
        "log_gain": torch.zeros(3),
        "scale": torch.ones((2, 3)),
        "shift": torch.zeros((2, 3)),
        "decoder.weight": torch.zeros((4, 1)),
        "decoder.bias": torch.zeros(4),
        "policy.weight": torch.zeros((1968, 4)),
        "policy.bias": torch.zeros(1968),
        "value.weight": torch.zeros((64, 4)),
        "value.bias": torch.zeros(64),
    }


def test_weights_parse_metadata_and_shapes(tmp_path):
    path = tmp_path / "w.safetensors"
    save_file(tensors(), path, metadata={"steps": "2", "alpha": "0.25", "hidden": "4", "step": "12000"})
    w = ChessFlyWeights.from_file(path)
    assert (w.steps, w.alpha, w.hidden, w.step) == (2, 0.25, 4, 12000)
    assert w.encoder_inputs == 1
    assert w.readout_neurons == 1


def test_weights_reject_missing_tensor():
    t = tensors(); t.pop("policy.bias")
    with pytest.raises(WeightFormatError, match="missing tensors"):
        ChessFlyWeights.from_tensors(t)


def test_weights_reject_incompatible_shape():
    t = tensors(); t["policy.weight"] = torch.zeros((10, 4))
    with pytest.raises(WeightFormatError, match="policy.weight"):
        ChessFlyWeights.from_tensors(t)
