import math
from dataclasses import dataclass

import torch

from chessfly_statepings.encoding import ACTION_INDEX
from chessfly_statepings.specificity import (
    centered_rms_delta,
    empirical_percentile,
    evaluate_directional_specificity,
    sign_symmetric,
)


def test_empirical_percentile_uses_midrank_for_ties():
    assert empirical_percentile(2.0, [1.0, 2.0, 3.0]) == 0.5
    assert empirical_percentile(4.0, [1.0, 2.0, 3.0]) == 0.875
    assert empirical_percentile(0.0, [1.0, 2.0, 3.0]) == 0.125


def test_centered_rms_ignores_common_logit_shift():
    base = torch.tensor([1.0, 2.0, 4.0])
    shifted = base + 10.0
    assert centered_rms_delta(base, shifted) == 0.0


def test_sign_symmetric_takes_larger_signed_effect():
    assert sign_symmetric(0.2, 0.8) == 0.8


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


@dataclass
class Trace:
    association: torch.Tensor
    policy_logits: torch.Tensor
    value_logits: torch.Tensor


@dataclass
class Forward:
    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    activity: tuple
    instability: str | None = None


class FakeModel:
    def __init__(self):
        self.readout_index = torch.tensor([0, 1, 2, 3])
        self.device = torch.device("cpu")
        self.trace_calls = []
        self.forward_calls = 0

    def _trace(self, readout):
        if readout.ndim == 1:
            readout = readout.unsqueeze(0)
        association = readout.clone()
        policy = torch.full((readout.shape[0], 1968), -5.0)
        policy[:, ACTION_INDEX["e2e4"]] = 2 * readout[:, 0] + readout[:, 2]
        policy[:, ACTION_INDEX["d2d4"]] = readout[:, 1] - readout[:, 3]
        value = torch.zeros((readout.shape[0], 64))
        value[:, 10] = readout[:, 0] - 2 * readout[:, 3]
        value[:, 20] = readout[:, 2] + readout[:, 1]
        return Trace(association, policy, value)

    def decode_readout_trace(self, readout):
        self.trace_calls.append(int(readout.shape[0] if readout.ndim > 1 else 1))
        return self._trace(readout)

    def forward(self, features, *, include_activity=False):
        self.forward_calls += 1
        activity = (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        )
        trace = self._trace(activity[-1])
        return Forward(trace.policy_logits, trace.value_logits, activity if include_activity else (), None)


def test_specificity_batches_controls_and_reports_correct_geometry():
    model = FakeModel()
    result = evaluate_directional_specificity(
        ["4k3/8/8/8/8/8/8/4K3 w - - 0 1"],
        model,
        rho=0.5,
        magnitudes=(1.0, 2.0),
        controls=4,
        seed=7,
        board_factory=Board,
    )
    assert model.forward_calls == 1
    # One baseline trace + one batched trace per magnitude. Each magnitude
    # contains (real + 4 controls) x (positive + negative) = 10 rows.
    assert model.trace_calls == [1, 10, 10]
    assert result.controls == 4
    assert len(result.runs) == 2
    row = result.runs[0].rows[0]
    assert row.orthogonal_norm_ratio > 0
    assert math.isclose(
        row.orthogonal_energy_fraction,
        row.orthogonal_norm_ratio**2,
        rel_tol=1e-6,
    )
    assert abs(row.orthogonal_cosine) < 1e-6
    for percentile in row.real_percentiles.values():
        assert 0 <= percentile <= 1
    assert row.real_plus.value_bin in range(64)
    assert row.real_minus.value_bin in range(64)
    assert result.runs[0].aggregate["positions"] == 1.0
