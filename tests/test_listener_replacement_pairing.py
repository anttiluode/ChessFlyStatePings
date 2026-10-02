import torch

from chessfly_statepings.listener_replacement import evaluate_listener_signatures


def test_identical_heads_use_paired_listener_replacements():
    memory = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    query = torch.tensor(
        [[0.8, 0.2, 0.1, 0.0], [0.1, 0.7, 0.2, 0.0], [0.1, 0.1, 0.8, 0.0]]
    )
    weight = torch.tensor(
        [[1.0, 2.0, -1.0, 0.5], [-0.5, 1.0, 2.0, -1.0]]
    )
    result = evaluate_listener_signatures(
        memory,
        query,
        value_weight=weight,
        policy_weight=weight,
        controls=1,
        seed=9,
    )
    assert result.value_control_mean == result.policy_control_mean
    assert result.value_percentiles == result.policy_percentiles
