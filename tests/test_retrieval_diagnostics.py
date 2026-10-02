import pytest
import torch

from chessfly_statepings.listener_replacement import apply_listener
from chessfly_statepings.retrieval_diagnostics import center_rows, retrieval_details


def test_center_rows_removes_common_shift():
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    shifted = values + torch.tensor([[10.0], [20.0]])
    assert torch.allclose(center_rows(values), center_rows(shifted), atol=1e-7)


def test_centered_listener_is_invariant_to_softmax_gauge_transform():
    signatures = torch.tensor([[1.0, -2.0, 0.5], [0.2, 1.4, -0.7]])
    weight = torch.tensor([[1.0, 0.0, 0.5], [-0.5, 1.0, 0.0], [0.2, -0.3, 0.8]])
    v = torch.tensor([0.7, -1.1, 0.4])
    gauge_shifted_weight = weight + torch.ones(3, 1) @ v.unsqueeze(0)

    original = center_rows(apply_listener(signatures, weight))
    shifted = center_rows(apply_listener(signatures, gauge_shifted_weight))
    assert torch.allclose(original, shifted, atol=1e-6)


def test_retrieval_details_reports_scores_ranks_and_margins():
    scores = torch.tensor(
        [[0.8, 0.2, 0.1], [0.4, 0.4, 0.3], [0.1, 0.2, 0.9]]
    )
    details = retrieval_details(scores)
    assert details.ranks == (1.0, 1.5, 1.0)
    assert details.margins[0] == pytest.approx(0.6)
    assert details.margins[1] == pytest.approx(0.0)
    assert details.scores[2][2] == pytest.approx(0.9)
