from types import SimpleNamespace

import torch
import torch.nn.functional as F

from chessfly_statepings.rectified_feedback import run_rectified_feedback


class _FakeModel:
    device = torch.device("cpu")
    weights = SimpleNamespace(steps=3, alpha=1.0)
    readout_index = torch.tensor([0, 1])
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).to_sparse()
    tensors = {
        "scale": torch.ones((3, 2)),
        "shift": torch.zeros((3, 2)),
        "decoder.weight": torch.eye(2),
        "decoder.bias": torch.zeros(2),
        "value.weight": torch.eye(2),
        "value.bias": torch.zeros(2),
        "policy.weight": torch.eye(2),
        "policy.bias": torch.zeros(2),
    }

    def _prepare(self, features):
        drive = torch.tensor([[1.0, 0.2]])
        return torch.zeros((1, 1)), drive

    def decode_readout_trace(self, values):
        association = F.gelu(values, approximate="tanh")
        return SimpleNamespace(
            association=association,
            value_logits=association,
            policy_logits=association,
        )


def test_zero_gamma_matches_linear_even_control():
    model = _FakeModel()
    baseline = run_rectified_feedback(
        model,
        torch.zeros(1),
        rho=0.75,
        magnitude=1.0,
        gamma=0.0,
        mode="nonlinear_even",
    )
    linear = run_rectified_feedback(
        model,
        torch.zeros(1),
        rho=0.75,
        magnitude=1.0,
        gamma=0.5,
        mode="linear_even",
    )
    assert torch.allclose(baseline.readout, linear.readout, atol=1e-7)


def test_nonlinear_even_feedback_changes_later_recurrent_state():
    model = _FakeModel()
    baseline = run_rectified_feedback(
        model,
        torch.zeros(1),
        rho=0.75,
        magnitude=1.0,
        gamma=0.0,
        mode="nonlinear_even",
    )
    feedback = run_rectified_feedback(
        model,
        torch.zeros(1),
        rho=0.75,
        magnitude=1.0,
        gamma=0.5,
        mode="nonlinear_even",
    )
    assert not torch.allclose(baseline.readout, feedback.readout)
    assert sum(feedback.feedback_norms) > 0.0
