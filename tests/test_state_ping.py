import torch

from chessfly_statepings.model import ChessFlyBaseline
from chessfly_statepings.state_ping import StatePingModel, trajectory_summary
from test_model import toy_graph, toy_weights


def test_trajectory_summary_has_analytical_fast_slow_residue():
    activity = (torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 1.0]]))
    summary = trajectory_summary(activity, rho=0.5, readout_indices=torch.tensor([1]))
    assert summary["steps"][0]["mean_abs_residue"] == 0.25
    assert summary["steps"][1]["mean_abs_residue"] == 0.375
    assert summary["steps"][1]["readout_mean_abs_residue"] == 0.5


def test_kappa_zero_matches_baseline():
    graph, weights = toy_graph(), toy_weights()
    x = torch.zeros(780); x[0] = 1
    baseline = ChessFlyBaseline(graph, weights).forward(x, include_activity=True)
    ping = StatePingModel(graph, weights).forward(x, rho=0.75, kappa=0.0, include_activity=True)
    assert ping.instability is None
    assert torch.allclose(ping.policy_logits, baseline.policy_logits, atol=0, rtol=0)
    assert torch.allclose(ping.value_logits, baseline.value_logits, atol=0, rtol=0)
    for a, b in zip(ping.activity, baseline.activity):
        assert torch.allclose(a, b, atol=0, rtol=0)


def test_positive_and_negative_kappa_change_recurrence_without_mutating_graph():
    graph, weights = toy_graph(), toy_weights()
    original_sign = graph.sign.copy(); original_cols = graph.col_idx.copy()
    x = torch.zeros(780); x[0] = 1
    pos = StatePingModel(graph, weights).forward(x, rho=0.5, kappa=0.2, include_activity=True)
    neg = StatePingModel(graph, weights).forward(x, rho=0.5, kappa=-0.2, include_activity=True)
    assert pos.activity[1][0,1] > 1.0
    assert neg.activity[1][0,1] < 1.0
    assert (graph.sign == original_sign).all()
    assert (graph.col_idx == original_cols).all()


def test_stateping_reports_nonfinite_activity():
    model = StatePingModel(toy_graph(), toy_weights(nan_scale=True))
    x = torch.zeros(780); x[0] = 1
    result = model.forward(x, rho=0.5, kappa=0.1)
    assert result.instability is not None
    assert result.policy_logits is None
