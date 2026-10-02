from types import SimpleNamespace

import torch

from chessfly_statepings.listener_replacement import (
    apply_listener,
    evaluate_listener_replacement,
    evaluate_listener_signatures,
    listener_retrieval,
    shuffled_listener_controls,
)


def test_apply_listener_is_linear_projection():
    signatures = torch.tensor([[1.0, 2.0, 3.0]])
    weight = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.5, 0.0]])
    assert torch.allclose(
        apply_listener(signatures, weight), torch.tensor([[-1.0, 1.0]])
    )


def test_shuffled_listener_controls_preserve_weight_spectrum():
    weight = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.5, 3.0]]
    )
    controls = shuffled_listener_controls(weight, controls=5, seed=7)
    base_sv = torch.linalg.svdvals(weight)
    assert len(controls) == 5
    for control in controls:
        assert torch.allclose(torch.linalg.svdvals(control), base_sv, atol=1e-6)
        assert not torch.equal(control, weight)


def test_listener_retrieval_uses_same_signatures_before_and_after_listener():
    memory = torch.tensor(
        [
            [1.5410, -0.2934, -2.1788, 0.5684],
            [-1.0845, -1.3986, 0.4033, 0.8380],
            [-0.7193, -0.4033, -0.5966, 0.1820],
        ]
    )
    query = torch.tensor(
        [
            [0.8557, 0.5871, -3.0357, 0.6666],
            [-1.5376, -1.1001, -0.3102, -0.3693],
            [-0.4229, 0.7619, 0.1552, 0.8019],
        ]
    )
    listener = torch.tensor(
        [[0.1919, 1.2638, -1.2904, -0.7911], [-0.0209, -0.7185, 0.5186, -1.3125]]
    )
    raw, heard = listener_retrieval(memory, query, listener)
    assert raw.accuracy < heard.accuracy
    assert heard.accuracy == 1.0
    assert heard.mean_correct_margin > raw.mean_correct_margin


def test_listener_signature_evaluation_holds_ping_fixed_and_replaces_listener():
    memory = torch.tensor(
        [
            [-0.2603556, 0.0509926, -1.1930875, -0.1355794, 0.2841277, -0.7730647],
            [0.0004432, -0.5773503, 1.0048155, -0.9464842, 0.7930469, -0.0264908],
            [-0.0784326, -0.0263984, -0.0501727, 0.5456939, 1.0036533, -0.6679118],
        ]
    )
    query = torch.tensor(
        [
            [-1.4089854, -2.3275180, -3.4623590, -0.1303105, 2.9665289, 1.0275027],
            [-0.1592505, -1.6397114, 0.3197919, -2.2425408, -1.9186370, -0.6059306],
            [1.4093204, 0.1573199, -2.1498923, 1.3185692, 1.0354575, 0.7076061],
        ]
    )
    value_weight = torch.tensor(
        [
            [0.5901633, 0.0091798, -0.2933321, -0.8725495, -0.4044339, 0.5561328],
            [0.3362199, 0.5766450, 0.2028129, 1.5771854, -0.0525458, -0.7984794],
            [2.7276034, -0.0928820, 0.6162329, -1.1290206, -1.4975572, -0.6628561],
        ]
    )
    policy_weight = torch.eye(6)
    result = evaluate_listener_signatures(
        memory,
        query,
        value_weight=value_weight,
        policy_weight=policy_weight,
        controls=32,
        seed=11,
    )
    assert result.association.accuracy < result.value.accuracy
    assert result.value.accuracy == 1.0
    assert result.value_control_mean.accuracy < result.value.accuracy
    assert result.value_percentiles["mean_correct_margin"] > 0.8
    assert result.policy.accuracy == result.association.accuracy


def test_evaluator_reuses_same_real_history_signature_across_listener_heads():
    class Model:
        device = torch.device("cpu")
        readout_index = torch.tensor([0, 1, 2])
        tensors = {
            "value.weight": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]]),
            "policy.weight": torch.eye(3),
        }

        def __init__(self):
            self.calls = 0

        def forward(self, features, include_activity=False):
            sign = 1.0 if self.calls == 0 else -1.0
            self.calls += 1
            activity = (
                torch.tensor([[1.0, sign, 0.0]]),
                torch.tensor([[1.5, 0.5 * sign, 0.2]]),
                torch.tensor([[2.0, 0.2 * sign, 0.4]]),
                torch.tensor([[2.5, 0.1 * sign, 0.7]]),
                torch.tensor([[2.8, 0.05 * sign, 1.0]]),
            )
            return SimpleNamespace(activity=activity, instability=None)

        def decode_readout_trace(self, readout):
            association = readout * readout
            value = association @ self.tensors["value.weight"].T
            policy = association @ self.tensors["policy.weight"].T
            return SimpleNamespace(
                association=association,
                value_logits=value,
                policy_logits=policy,
            )

    fens = [
        "8/8/8/8/8/8/8/K6k w - - 0 1",
        "8/8/8/8/8/8/7P/K6k w - - 0 1",
    ]
    result = evaluate_listener_replacement(
        fens,
        Model(),
        rho=0.75,
        magnitudes=(1.0, 2.0),
        controls=3,
        seed=4,
    )
    assert result.positions == 2
    assert result.controls == 3
    assert len(result.runs) == 2
    for run in result.runs:
        assert run.magnitude in (1.0, 2.0)
        assert hasattr(run, "association")
        assert hasattr(run, "value")
        assert hasattr(run, "policy")
