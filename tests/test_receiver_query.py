import torch

from chessfly_statepings.receiver_query import receiver_signature


def test_receiver_signature_is_symmetric_directional_effect():
    class FakeTrace:
        def __init__(self, association, value_logits):
            self.association = association
            self.value_logits = value_logits

    class FakeModel:
        def decode_readout_trace(self, readout):
            association = readout * readout
            value = torch.stack(
                [association.sum(dim=1), association[:, 0] - association[:, 1]], dim=1
            )
            return FakeTrace(association, value)

    current = torch.tensor([[2.0, 3.0]])
    direction = torch.tensor([[1.0, -2.0]])
    association, value = receiver_signature(FakeModel(), current, direction, magnitude=0.5)

    expected_association = 2.0 * current * direction
    expected_value = torch.stack(
        [expected_association.sum(dim=1), expected_association[:, 0] - expected_association[:, 1]], dim=1
    )
    assert torch.allclose(association, expected_association)
    assert torch.allclose(value, expected_value)


def test_receiver_geometry_can_recover_identity_that_raw_history_misses():
    from chessfly_statepings.receiver_query import evaluate_receiver_geometry

    class FakeTrace:
        def __init__(self, association, value_logits):
            self.association = association
            self.value_logits = value_logits

    class FakeModel:
        def decode_readout_trace(self, readout):
            association = readout * readout
            value = torch.stack(
                [association.sum(dim=1), association[:, 0] - association[:, 1]], dim=1
            )
            return FakeTrace(association, value)

    current = torch.tensor([
        [3.0922, 27.4063, 0.9214, 0.5066],
        [0.9613, 0.0857, 0.7469, 8.0476],
        [0.9786, 0.2558, 5.9584, 0.2122],
    ])
    memory_history = torch.tensor([
        [0.1403, 0.0176, -0.5389, -0.8304],
        [-0.5016, 0.8635, 0.0219, 0.0487],
        [-0.1420, -0.8225, 0.0391, 0.5493],
    ])
    query_history = torch.tensor([
        [0.2959, 0.0018, -0.6716, -0.6793],
        [0.1537, -0.1573, -0.9728, 0.0736],
        [-0.7530, 0.4688, 0.1194, -0.4461],
    ])

    result = evaluate_receiver_geometry(
        FakeModel(),
        memory_current=current,
        memory_history=memory_history,
        query_current=current,
        query_history=query_history,
        magnitudes=(1.0,),
        controls=32,
        seed=0,
    )

    assert result.raw_history.accuracy < 1.0
    assert result.present_receiver.accuracy == 1.0
    run = result.runs[0]
    assert run.association.accuracy == 1.0
    assert run.association.mean_correct_margin > 0.0
    assert run.association_control_mean.accuracy < run.association.accuracy
    assert run.association_percentiles["mean_correct_margin"] > 0.9


def test_evaluator_builds_step4_memory_and_step5_receiver_queries():
    from chessfly_statepings.receiver_query import evaluate_receiver_query

    class FakeTrace:
        def __init__(self, association, value_logits):
            self.association = association
            self.value_logits = value_logits

    class FakeResult:
        def __init__(self, activity):
            self.activity = activity
            self.instability = None

    class FakeModel:
        device = torch.device("cpu")
        readout_index = torch.tensor([0, 1, 2])

        def __init__(self):
            self.calls = 0

        def forward(self, features, include_activity=False):
            s = 1.0 if self.calls == 0 else -1.0
            self.calls += 1
            activity = (
                torch.tensor([[1.0, s, 0.0]]),
                torch.tensor([[1.5, 0.5 * s, 0.2]]),
                torch.tensor([[2.0, 0.2 * s, 0.4]]),
                torch.tensor([[2.5, 0.1 * s, 0.7]]),
                torch.tensor([[2.8, 0.05 * s, 1.0]]),
            )
            return FakeResult(activity)

        def decode_readout_trace(self, readout):
            association = torch.stack(
                [readout[:, 0] ** 2, readout[:, 1] ** 2, readout[:, 2] ** 2], dim=1
            )
            value = torch.stack([association.sum(dim=1), association[:, 0] - association[:, 2]], dim=1)
            return FakeTrace(association, value)

    fens = [
        "8/8/8/8/8/8/8/K6k w - - 0 1",
        "8/8/8/8/8/8/7P/K6k w - - 0 1",
    ]
    model = FakeModel()
    result = evaluate_receiver_query(
        fens, model, rho=0.75, magnitudes=(1.0, 2.0), controls=3, seed=4
    )

    assert model.calls == 2
    assert result.positions == 2
    assert result.controls == 3
    assert result.seed == 4
    assert len(result.runs) == 2
