from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chessfly_statepings.graph import ChessFlyGraph
from chessfly_statepings.model import ChessFlyBaseline
from chessfly_statepings.weights import ChessFlyWeights


def memory_model():
    """Two independent leaky states: h_next = (h + input)/2."""
    graph = ChessFlyGraph(
        node_count=2,
        row_ptr=np.array([0, 1, 2], dtype=np.int64),
        col_idx=np.array([0, 1], dtype=np.int64),
        sign=np.ones(2, dtype=np.int16),
        groups=np.array([2, 3], dtype=np.uint8),
        inputs=np.array([0, 1], dtype=np.int64),
        readout=np.array([0, 1], dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
    )
    encoder = torch.zeros((2, 780))
    encoder[:, :2] = torch.eye(2)
    tensors = {
        "encoder.weight": encoder, "encoder.bias": torch.zeros(2),
        "log_gain": torch.zeros(2), "scale": torch.full((5, 2), 0.5),
        "shift": torch.zeros((5, 2)), "decoder.weight": torch.eye(2),
        "decoder.bias": torch.zeros(2),
        "policy.weight": torch.zeros((1968, 2)), "policy.bias": torch.zeros(1968),
        "value.weight": torch.zeros((64, 2)), "value.bias": torch.zeros(64),
    }
    weights = ChessFlyWeights.from_tensors(tensors, {"steps": "5", "alpha": "1", "hidden": "2"})
    return ChessFlyBaseline(graph, weights, device="cpu")


def ordered_schedule():
    x = torch.zeros((5, 2, 780))
    x[0, :, :2] = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    x[1, :, :2] = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    x[2:, :, :2] = 1.0
    return x


def test_ordered_histories_survive_identical_present_then_decay():
    from chessfly_statepings.history_lens import scheduled_activity

    states = scheduled_activity(memory_model(), ordered_schedule())
    assert torch.equal(states[2], torch.tensor([[0.75, 1.0], [1.0, 0.75]]))
    assert torch.equal(states[4], torch.tensor([[0.9375, 1.0], [1.0, 0.9375]]))


def test_constant_schedule_reproduces_existing_baseline():
    from chessfly_statepings.history_lens import scheduled_activity

    model = memory_model()
    x = torch.zeros((2, 780)); x[:, :2] = 1.0
    result = scheduled_activity(model, x.unsqueeze(0).repeat(5, 1, 1))
    baseline = model.forward(x, include_activity=True)
    assert all(torch.equal(a, b) for a, b in zip(result, baseline.activity))


def test_future_input_changes_cannot_change_earlier_receiver_states():
    from chessfly_statepings.history_lens import scheduled_activity

    original = ordered_schedule(); changed = original.clone(); changed[4, :, :2] = 10
    before = scheduled_activity(memory_model(), original)
    after = scheduled_activity(memory_model(), changed)
    assert all(torch.equal(a, b) for a, b in zip(before[:4], after[:4]))
    assert not torch.equal(before[4], after[4])


def test_pair_score_rewards_correct_continuations_and_penalizes_swap():
    from chessfly_statepings.history_lens import paired_continuation_scores

    target = torch.eye(2)
    native = paired_continuation_scores(torch.eye(2), target)
    swapped = paired_continuation_scores(torch.eye(2).flip(0), target)
    assert native["scores"] == [[1.0, 0.0], [0.0, 1.0]]
    assert native["accuracy"] == 1.0
    assert native["mean_correct_margin"] == 1.0
    assert swapped["accuracy"] == 0.0
    assert swapped["mean_correct_margin"] == -1.0


def test_state_blind_reader_gets_half_credit_and_no_native_margin():
    from chessfly_statepings.history_lens import paired_continuation_scores

    result = paired_continuation_scores(torch.ones((2, 2)), torch.eye(2))
    assert result["accuracy"] == 0.5
    assert result["mean_correct_margin"] == 0.0
    assert result["response_separation"] == 0.0


@pytest.mark.parametrize("targets", [torch.zeros((2, 2)), torch.ones((2, 2))])
def test_unidentifiable_targets_are_not_counted_as_success(targets):
    from chessfly_statepings.history_lens import paired_continuation_scores

    result = paired_continuation_scores(torch.eye(2), targets)
    assert result["identifiable"] is False
    assert result["accuracy"] is None


def test_schedule_rejects_invalid_length_and_nonfinite_input():
    from chessfly_statepings.history_lens import scheduled_activity

    with pytest.raises(ValueError, match="steps"):
        scheduled_activity(memory_model(), ordered_schedule()[:4])
    x = ordered_schedule(); x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        scheduled_activity(memory_model(), x)


def test_fixed_ping_has_identical_content_and_state_specific_useful_effects():
    from chessfly_statepings.history_lens import fixed_ping_association, paired_continuation_scores

    class SquareReceiver:
        def decode_readout_trace(self, x):
            return SimpleNamespace(association=x.square())

    states = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    ping = torch.ones((1, 2))
    response = fixed_ping_association(SquareReceiver(), states, ping, magnitude=0.5)
    # Hand-derived: derivative of x^2 times p, with the same p for both.
    assert torch.equal(response, torch.tensor([[2.0, 4.0], [4.0, 2.0]]))
    # Independent continuation: doubling the latent state changes x^2 by 3*x^2.
    score = paired_continuation_scores(response, torch.tensor([[3.0, 12.0], [12.0, 3.0]]))
    assert score["accuracy"] == 1.0
    assert score["mean_correct_margin"] > 0.3


def test_pair_schedule_uses_same_cue_bag_and_identical_present_suffix():
    from chessfly_statepings.history_lens import paired_history_activity

    cues = torch.zeros((2, 780)); cues[:, :2] = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    present = torch.zeros(780); present[:2] = 1.0
    states = paired_history_activity(memory_model(), cues, present)
    assert torch.equal(states[1], torch.tensor([[0.5, 1.0], [1.0, 0.5], [0.75, 0.75]]))
    assert torch.equal(states[2], torch.tensor([[0.75, 1.0], [1.0, 0.75], [0.875, 0.875]]))


@pytest.mark.parametrize("prefix_steps, expected", [(2, 0.09375), (3, 0.0703125)])
def test_ping_uses_only_reference_prefix_not_receiver_history_or_future(prefix_steps, expected):
    from chessfly_statepings.history_lens import fixed_history_ping

    model = memory_model()
    states = [torch.zeros((3, 2)) for _ in range(5)]
    states[0][2] = torch.tensor([1.0, 0.0])
    states[1][2] = torch.tensor([1.0, 1.0])
    states[2][2] = torch.tensor([1.0, 1.0])
    before = fixed_history_ping(tuple(states), model, rho=0.75, prefix_steps=prefix_steps)
    for state in states[:prefix_steps]:
        state[:2] = 40
    states[prefix_steps:] = [torch.full((3, 2), 100.0) for _ in states[prefix_steps:]]
    after = fixed_history_ping(tuple(states), model, rho=0.75, prefix_steps=prefix_steps)
    assert torch.equal(before.orthogonal, after.orthogonal)
    assert torch.allclose(before.orthogonal, torch.tensor([[-expected, expected]]))


def test_default_ping_waits_for_history_after_zero_first_readout():
    from chessfly_statepings.history_lens import fixed_history_ping

    model = memory_model()
    states = [torch.zeros((3, 2)) for _ in range(5)]
    states[1][2] = torch.tensor([1.0, 0.0])
    states[2][2] = torch.tensor([1.0, 1.0])
    ping = fixed_history_ping(tuple(states), model)
    assert torch.allclose(ping.orthogonal, torch.tensor([[-0.09375, 0.09375]]))


def test_ping_must_be_constructed_before_delayed_query():
    from chessfly_statepings.history_lens import evaluate_history_lens, load_history_cases
    from pathlib import Path

    cases = load_history_cases(Path(__file__).resolve().parents[1] / "data" / "history_lens_cases.json")[:1]
    with pytest.raises(ValueError, match="after ping construction"):
        evaluate_history_lens(cases, memory_model(), delays=(1,), ping_steps=3)


def test_case_file_rejects_duplicates_and_encoding_identical_cues(tmp_path):
    import json
    from chessfly_statepings.history_lens import load_history_cases

    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    a = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"
    b = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1"
    row = {"id": "one", "present": start, "cue_a": a, "cue_b": b}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [row, row]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_history_cases(path)
    row["cue_b"] = a.replace("1 1", "2 2")
    path.write_text(json.dumps({"cases": [row]}))
    with pytest.raises(ValueError, match="encodings"):
        load_history_cases(path)


def test_real_tiny_model_zero_ping_remains_inconclusive_and_finite():
    import json
    from chessfly_statepings.history_lens import load_history_cases, evaluate_history_lens

    cases = load_history_cases("data/history_lens_cases.json")[:1]
    result = evaluate_history_lens(cases, memory_model(), controls=2)
    assert result["primary"]["status"] == "inconclusive"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"delays": [0]}, {"delays": [3]}, {"magnitudes": [float("nan")]},
    {"magnitudes": [float("inf")]}, {"rho": float("nan")}, {"controls": 0},
])
def test_invalid_protocol_parameters_fail_explicitly(kwargs):
    from chessfly_statepings.history_lens import load_history_cases, evaluate_history_lens

    cases = load_history_cases("data/history_lens_cases.json")[:1]
    with pytest.raises(ValueError):
        evaluate_history_lens(cases, memory_model(), **kwargs)
