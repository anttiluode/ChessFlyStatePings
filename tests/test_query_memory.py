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
