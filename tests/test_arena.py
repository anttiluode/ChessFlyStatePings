from dataclasses import dataclass

from chessfly_statepings.arena import play_paired_arena
from chessfly_statepings.policy import PolicyInstabilityError


class Move:
    def __init__(self, uci): self._uci=uci
    def uci(self): return self._uci


@dataclass
class Decision:
    move: Move
    selected_uci: str


class Policy:
    def __init__(self, uci, fail=False): self.uci=uci; self.fail=fail
    def select(self, board):
        if self.fail: raise PolicyInstabilityError("unstable")
        return Decision(Move(self.uci), self.uci)


class Board:
    def __init__(self, _fen, terminal_after=2, fixed_result="1-0"):
        self.turn=True; self.plies=0; self.terminal_after=terminal_after; self.fixed_result=fixed_result
    def is_game_over(self, claim_draw=True): return self.plies >= self.terminal_after
    def result(self, claim_draw=True): return self.fixed_result if self.is_game_over() else "*"
    def push(self, move): self.plies += 1; self.turn = not self.turn
    def fen(self): return f"toy-{self.plies}"


def test_paired_colors_make_fixed_white_win_symmetric_between_policies():
    result=play_paired_arena(Policy("a2a3"),Policy("b2b3"),openings=["x"],games=2,max_plies=4,board_factory=Board,seed=7)
    assert result.baseline_wins == 1
    assert result.stateping_wins == 1
    assert result.draws == 0
    assert result.baseline_score == 0.5
    assert result.move_disagreement_rate == 1.0
    assert result.mean_game_length == 2.0


def test_max_ply_is_recorded_as_draw():
    factory=lambda fen: Board(fen,terminal_after=99)
    result=play_paired_arena(Policy("a2a3"),Policy("a2a3"),openings=["x"],games=2,max_plies=2,board_factory=factory)
    assert result.draws == 2
    assert result.forfeits == 0


def test_policy_instability_forfeits_for_that_policy():
    factory=lambda fen: Board(fen,terminal_after=10)
    result=play_paired_arena(Policy("a2a3"),Policy("b2b3",fail=True),openings=["x"],games=2,max_plies=4,board_factory=factory)
    assert result.baseline_wins == 2
    assert result.forfeits == 2
