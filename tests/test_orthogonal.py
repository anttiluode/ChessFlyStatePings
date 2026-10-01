import torch

from chessfly_statepings.model import ChessFlyBaseline, ForwardResult
from chessfly_statepings.orthogonal import (
    evaluate_orthogonal_positions,
    orthogonal_component,
    orthogonal_history_readout,
    shuffled_like,
    shuffled_orthogonal_control,
)
from chessfly_statepings.encoding import ACTION_INDEX
from test_model import toy_graph, toy_weights


def test_orthogonal_component_removes_current_direction():
    h = torch.tensor([[1.0, 0.0]])
    residue = torch.tensor([[1.0, 1.0]])
    perpendicular, ratio, cosine = orthogonal_component(h, residue)
    assert torch.allclose(perpendicular, torch.tensor([[0.0, 1.0]]), atol=1e-7)
    assert torch.allclose(ratio, torch.tensor([2 ** -0.5]), atol=1e-7)
    assert torch.allclose(cosine, torch.zeros(1), atol=1e-7)


def test_zero_present_state_leaves_residue_unchanged():
    h = torch.zeros((1, 2))
    residue = torch.tensor([[2.0, -1.0]])
    perpendicular, ratio, cosine = orthogonal_component(h, residue)
    assert torch.equal(perpendicular, residue)
    assert torch.allclose(ratio, torch.ones(1))
    assert torch.allclose(cosine, torch.zeros(1))


def test_history_readout_uses_final_readout_subspace():
    activity = (
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[1.0, 2.0]]),
    )
    state = orthogonal_history_readout(activity, rho=0.5, readout_indices=torch.tensor([1]))
    assert state.current.shape == (1, 1)
    assert torch.allclose(state.current, torch.tensor([[2.0]]))
    assert state.orthogonal.shape == (1, 1)
    assert torch.allclose(state.orthogonal, torch.zeros((1, 1)), atol=1e-6)


def test_shuffle_is_deterministic_and_norm_preserving():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    first = shuffled_like(values, seed=7)
    second = shuffled_like(values, seed=7)
    assert torch.equal(first, second)
    assert torch.allclose(torch.linalg.vector_norm(first, dim=1), torch.linalg.vector_norm(values, dim=1))
    assert not torch.equal(first, values)


def test_shuffled_orthogonal_control_stays_orthogonal_and_energy_matched():
    h = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    orthogonal = torch.tensor([[2.0, -1.0, 0.0, 0.0]])
    control = shuffled_orthogonal_control(h, orthogonal, seed=3)
    assert abs(float(torch.sum(h * control).item())) < 1e-6
    assert torch.allclose(
        torch.linalg.vector_norm(control, dim=1),
        torch.linalg.vector_norm(orthogonal, dim=1),
        atol=1e-6,
    )
    assert not torch.equal(control, orthogonal)


def test_decode_readout_matches_baseline_final_decode():
    model = ChessFlyBaseline(toy_graph(), toy_weights())
    x = torch.zeros(780)
    x[0] = 1.0
    baseline = model.forward(x, include_activity=True)
    readout = baseline.activity[-1].index_select(1, model.readout_index)
    decoded = model.decode_readout(readout)
    assert torch.allclose(decoded.policy_logits, baseline.policy_logits, atol=0, rtol=0)
    assert torch.allclose(decoded.value_logits, baseline.value_logits, atol=0, rtol=0)


class Move:
    def __init__(self, uci):
        self._uci = uci

    def uci(self):
        return self._uci


class Board:
    def __init__(self, fen):
        self._fen = fen
        self.legal_moves = (Move("e2e4"), Move("d2d4"))

    def fen(self):
        return self._fen


class FakeDecodeModel:
    def __init__(self):
        self.readout_index = torch.tensor([0, 1, 2])
        self.device = torch.device("cpu")

    def forward(self, features, *, include_activity=False):
        activity = (
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 1.0]]),
        )
        return self.decode_readout(activity[-1], activity=activity if include_activity else ())

    def decode_readout(self, readout, *, activity=()):
        policy = torch.full((1, 1968), -10.0)
        policy[0, ACTION_INDEX["e2e4"]] = readout[0, 0]
        policy[0, ACTION_INDEX["d2d4"]] = readout[0, 2]
        value = torch.zeros((1, 64))
        return ForwardResult(policy, value, tuple(activity), None)


def test_evaluate_orthogonal_positions_distinguishes_real_from_shuffled_direction():
    result = evaluate_orthogonal_positions(
        ["4k3/8/8/8/8/8/8/4K3 w - - 0 1"],
        FakeDecodeModel(),
        rho=0.5,
        lambdas=(2.0,),
        seed=0,
        board_factory=Board,
    )
    run = result.runs[0]
    assert run.aggregate["positions"] == 1.0
    assert run.aggregate["mean_orthogonal_energy_ratio"] > 0
    assert run.aggregate["mean_abs_orthogonal_cosine"] < 1e-6
    assert run.aggregate["real_move_change_rate"] == 0.0
    assert run.aggregate["shuffled_move_change_rate"] == 1.0


class StreamingModel(FakeDecodeModel):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, features, *, include_activity=False):
        self.events.append("forward")
        activity = (
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 1.0]]),
        )
        policy = torch.full((1, 1968), -10.0)
        policy[0, ACTION_INDEX["e2e4"]] = 1.0
        policy[0, ACTION_INDEX["d2d4"]] = 1.0
        value = torch.zeros((1, 64))
        return ForwardResult(policy, value, activity if include_activity else (), None)

    def decode_readout(self, readout, *, activity=()):
        self.events.append("decode")
        return super().decode_readout(readout, activity=activity)


def test_evaluator_streams_positions_before_requesting_the_next_fen():
    events = []
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"

    def positions():
        yield fen
        assert events.count("decode") >= 2
        yield fen

    evaluate_orthogonal_positions(
        positions(),
        StreamingModel(events),
        rho=0.5,
        lambdas=(1.0,),
        seed=0,
        board_factory=Board,
    )
    assert events.count("forward") == 2
