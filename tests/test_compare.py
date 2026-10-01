import pytest
from chessfly_statepings.compare import compare_positions
from chessfly_statepings.policy import PolicyCandidate, PolicyDecision


class Board:
    def __init__(self, fen): self._fen=fen
    def fen(self): return self._fen


class P:
    def __init__(self, choice, probs, value): self.choice=choice; self.probs=probs; self.value=value; self.calls=[]
    def select(self, board):
        self.calls.append(board.fen())
        return PolicyDecision(None,self.choice,(PolicyCandidate(self.choice,self.probs[self.choice],self.choice),),self.probs,self.value,{"active_fraction":0.5})


def test_compare_positions_is_paired_and_reports_divergence_and_move_change():
    fens=["fen-a","fen-b"]
    a=P("e2e4", {"e2e4":0.8,"d2d4":0.2}, 0.4)
    b=P("d2d4", {"e2e4":0.3,"d2d4":0.7}, 0.6)
    result=compare_positions(fens,a,b,board_factory=Board)
    assert a.calls == fens and b.calls == fens
    assert result.rows[0].move_changed is True
    assert result.rows[0].js_divergence > 0
    assert result.rows[0].value_delta == pytest.approx(0.2)
    assert result.aggregate["move_change_rate"] == 1.0


def test_compare_does_not_require_stockfish():
    result=compare_positions(["x"],P("a",{"a":1.0},0.5),P("a",{"a":1.0},0.5),board_factory=Board)
    assert result.rows[0].engine is None


class UnstableP:
    def select(self, board):
        from chessfly_statepings.policy import PolicyInstabilityError
        raise PolicyInstabilityError("non-finite activity")


def test_compare_records_stateping_instability_instead_of_aborting():
    baseline=P("a",{"a":1.0},0.5)
    result=compare_positions(["x"],baseline,UnstableP(),board_factory=Board)
    assert result.rows[0].stateping_move is None
    assert result.rows[0].instability == "stateping: non-finite activity"
    assert result.aggregate["instability_count"] == 1.0
    assert result.aggregate["valid_positions"] == 0.0
