from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from chessfly_statepings.curvature import (
    analytic_gelu_tanh_even_response,
    even_receiver_response,
)


class _LinearModel:
    def decode_readout_trace(self, values):
        return SimpleNamespace(association=2.0 * values + 3.0)


def test_linear_decoder_has_zero_even_response():
    current = torch.tensor([[1.0, 2.0]])
    direction = torch.tensor([[0.3, -0.7]])
    response = even_receiver_response(
        _LinearModel(), current, direction, magnitude=2.0
    )
    assert torch.allclose(response, torch.zeros_like(response), atol=1e-6)


def test_analytic_gelu_curvature_matches_small_finite_difference():
    current = torch.tensor([[0.2, -0.4, 0.7]])
    direction = torch.tensor([[0.3, 0.1, -0.2]])
    weight = torch.tensor([[0.8, -0.4, 0.2], [-0.1, 0.5, 0.7]])
    bias = torch.tensor([0.05, -0.2])

    class Model:
        def decode_readout_trace(self, values):
            association = F.gelu(values @ weight.T + bias, approximate="tanh")
            return SimpleNamespace(association=association)

    magnitude = 1e-2
    exact = even_receiver_response(
        Model(), current, direction, magnitude=magnitude
    )
    predicted = analytic_gelu_tanh_even_response(
        current,
        direction,
        weight,
        bias,
        magnitude=magnitude,
    )
    assert torch.allclose(exact, predicted, rtol=5e-3, atol=1e-7)


def test_even_response_scales_quadratically_at_small_amplitude():
    current = torch.tensor([[0.2, -0.4, 0.7]])
    direction = torch.tensor([[0.3, 0.1, -0.2]])
    weight = torch.tensor([[0.8, -0.4, 0.2], [-0.1, 0.5, 0.7]])
    bias = torch.tensor([0.05, -0.2])

    class Model:
        def decode_readout_trace(self, values):
            association = F.gelu(values @ weight.T + bias, approximate="tanh")
            return SimpleNamespace(association=association)

    small = even_receiver_response(Model(), current, direction, magnitude=0.01)
    doubled = even_receiver_response(Model(), current, direction, magnitude=0.02)
    ratio = torch.linalg.vector_norm(doubled) / torch.linalg.vector_norm(small)
    assert float(ratio) == pytest.approx(4.0, rel=0.03)
