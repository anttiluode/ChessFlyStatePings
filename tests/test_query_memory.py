import torch

from chessfly_statepings.query_memory import retrieval_metrics, score_queries


def test_history_coordinate_disambiguates_same_present_queries():
    present_q = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    present_k = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    history_q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    history_k = history_q.clone()

    present_scores = score_queries(present_q, present_k)
    history_scores = score_queries(present_q, present_k, history_q, history_k)

    assert torch.allclose(present_scores, torch.ones_like(present_scores))
    assert torch.argmax(history_scores, dim=1).tolist() == [0, 1]

    present = retrieval_metrics(present_scores)
    history = retrieval_metrics(history_scores)
    assert history.accuracy > present.accuracy
    assert history.mean_reciprocal_rank > present.mean_reciprocal_rank


def test_shuffled_history_loses_identity_specific_advantage():
    present_q = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    present_k = present_q.clone()
    history_q = torch.eye(3)
    history_k = history_q.clone()

    real = retrieval_metrics(score_queries(present_q, present_k, history_q, history_k))
    shuffled = retrieval_metrics(
        score_queries(present_q, present_k, history_q.roll(1, dims=0), history_k)
    )

    assert real.accuracy == 1.0
    assert shuffled.accuracy < real.accuracy
    assert shuffled.mean_correct_margin < real.mean_correct_margin


def test_evaluator_uses_previous_settle_as_memory_and_final_ping_as_query():
    from types import SimpleNamespace
    from chessfly_statepings.query_memory import evaluate_query_memory

    class FakeModel:
        device = torch.device("cpu")
        readout_index = torch.tensor([0, 1])

        def __init__(self):
            self.calls = 0

        def forward(self, features, include_activity=False):
            sign = 1.0 if self.calls == 0 else -1.0
            self.calls += 1
            activity = (
                torch.tensor([[1.0, sign]]),
                torch.tensor([[1.0, sign]]),
                torch.tensor([[1.0, sign]]),
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[1.0, 0.0]]),
            )
            return SimpleNamespace(activity=activity, instability=None)

    fens = [
        "8/8/8/8/8/8/8/K6k w - - 0 1",
        "8/8/8/8/8/8/7P/K6k w - - 0 1",
    ]
    result = evaluate_query_memory(fens, FakeModel(), rho=0.75, seed=0)
    assert result.positions == 2
    assert result.history.accuracy == 1.0
    assert result.history.mean_correct_margin > result.present.mean_correct_margin
    assert result.shuffled_history.mean_correct_margin < result.history.mean_correct_margin
