from types import SimpleNamespace

import pytest
import torch

from chessfly_statepings.state_crossing import cross_odd_responses, diagonal_advantage


class _SquareModel:
    def decode_readout_trace(self, values):
        return SimpleNamespace(association=values * values, value_logits=values * values)


class _LinearModel:
    def decode_readout_trace(self, values):
        return SimpleNamespace(association=2.0 * values, value_logits=2.0 * values)


def test_nonlinear_receiver_makes_same_ping_state_dependent():
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pings = torch.tensor([[0.5, -0.25], [0.2, 0.1]])
    responses = cross_odd_responses(
        _SquareModel(), states, pings, magnitude=0.1
    )
    assert responses.shape == (2, 2, 2)
    assert not torch.allclose(responses[0, 0], responses[1, 0])


def test_linear_receiver_response_is_state_blind():
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pings = torch.tensor([[0.5, -0.25], [0.2, 0.1]])
    responses = cross_odd_responses(
        _LinearModel(), states, pings, magnitude=0.1
    )
    assert torch.allclose(responses[0, 0], responses[1, 0], atol=1e-5)
    assert torch.allclose(responses[0, 1], responses[1, 1], atol=1e-5)


def test_diagonal_advantage_separates_native_from_crossed_pairs():
    matrix = torch.tensor(
        [[0.9, 0.1, 0.2], [0.3, 0.8, 0.2], [0.1, 0.2, 0.7]]
    )
    diagonal, off_diagonal, advantage = diagonal_advantage(matrix)
    assert float(diagonal) == pytest.approx(0.8)
    assert float(off_diagonal) == pytest.approx(
        (0.1 + 0.2 + 0.3 + 0.2 + 0.1 + 0.2) / 6
    )
    assert float(advantage) > 0.5
