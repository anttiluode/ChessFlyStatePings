import torch
import pytest

from chessfly_statepings.encoding import ACTION_INDEX
from chessfly_statepings.model import ForwardResult
from chessfly_statepings.policy import ChessFlyPolicy, PolicyInstabilityError


class Move:
    def __init__(self, uci): self._uci = uci
    def uci(self): return self._uci


class Board:
    def __init__(self, fen, moves): self._fen=fen; self.legal_moves=tuple(Move(m) for m in moves)
    def fen(self): return self._fen
    def san(self, move): return move.uci()


class Model:
    def __init__(self, best, *, unstable=False): self.best=best; self.unstable=unstable; self.graph=type('G',(),{'readout':[0]})()
    def forward(self, features, include_activity=False, **kwargs):
        if self.unstable: return ForwardResult(None, None, (), "boom")
        p=torch.full((1,1968), -10.0); p[0,ACTION_INDEX[self.best]]=5.0
        v=torch.zeros((1,64)); v[0,47]=3.0
        act=(torch.tensor([[1.0]]),) if include_activity else ()
        return ForwardResult(p,v,act,None)


def test_policy_masks_to_legal_moves_and_returns_top_probabilities():
    board=Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1", ["e2e4","d2d4"])
    decision=ChessFlyPolicy(Model("e2e4")).select(board)
    assert decision.selected_uci == "e2e4"
    assert set(decision.legal_probabilities) == {"e2e4","d2d4"}
    assert decision.top[0].uci == "e2e4"
    assert 0 <= decision.value_expectation <= 1


def test_black_to_move_uses_mirrored_action_indices_and_returns_original_move():
    board=Board("4k3/8/8/8/8/8/8/4K3 b - - 0 1", ["e7e5","d7d5"])
    decision=ChessFlyPolicy(Model("e2e4")).select(board)
    assert decision.selected_uci == "e7e5"


def test_policy_refuses_unstable_forward():
    board=Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1", ["e2e4"])
    with pytest.raises(PolicyInstabilityError, match="boom"):
        ChessFlyPolicy(Model("e2e4", unstable=True)).select(board)
