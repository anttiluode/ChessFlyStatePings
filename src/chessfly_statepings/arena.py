"""Deterministic paired-color headless chess arena."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .policy import PolicyError


@dataclass(frozen=True, slots=True)
class ArenaGame:
    opening: str
    white: str
    black: str
    result: str
    plies: int
    forfeit: str | None = None


@dataclass(frozen=True, slots=True)
class ArenaResult:
    games: tuple[ArenaGame, ...]
    baseline_wins: int
    stateping_wins: int
    draws: int
    baseline_score: float
    stateping_score: float
    mean_game_length: float
    move_disagreement_rate: float
    forfeits: int


def _default_board_factory(fen: str):
    try:
        import chess
    except ImportError as exc:
        raise RuntimeError("python-chess is required for headless games") from exc
    return chess.Board(fen)


def _result_for_forfeit(board: Any) -> str:
    return "0-1" if bool(board.turn) else "1-0"


def _play_game(
    opening: str,
    white_name: str,
    black_name: str,
    policies: dict[str, Any],
    *,
    max_plies: int,
    board_factory: Callable[[str], Any],
) -> tuple[ArenaGame, int, int]:
    board = board_factory(opening)
    disagreements = 0
    compared = 0
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        current_name = white_name if bool(board.turn) else black_name
        other_name = black_name if bool(board.turn) else white_name
        try:
            decision = policies[current_name].select(board)
        except Exception as exc:
            if not isinstance(exc, PolicyError):
                raise
            result = _result_for_forfeit(board)
            return ArenaGame(opening, white_name, black_name, result, plies, current_name), disagreements, compared
        try:
            other = policies[other_name].select(board)
        except Exception as exc:
            if not isinstance(exc, PolicyError):
                raise
        else:
            compared += 1
            disagreements += int(other.selected_uci != decision.selected_uci)
        board.push(decision.move)
        plies += 1
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
    else:
        result = "1/2-1/2"
    return ArenaGame(opening, white_name, black_name, result, plies, None), disagreements, compared


def play_paired_arena(
    baseline_policy: Any,
    stateping_policy: Any,
    *,
    openings: Sequence[str],
    games: int = 20,
    max_plies: int = 300,
    seed: int = 0,
    board_factory: Callable[[str], Any] | None = None,
) -> ArenaResult:
    if games < 2 or games % 2:
        raise ValueError("games must be a positive even number so colors are paired")
    if not openings:
        raise ValueError("at least one opening FEN is required")
    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    rng = random.Random(seed)
    factory = board_factory or _default_board_factory
    policies = {"baseline": baseline_policy, "stateping": stateping_policy}
    records: list[ArenaGame] = []
    disagreements = compared = 0
    for _ in range(games // 2):
        opening = openings[rng.randrange(len(openings))]
        for white, black in (("baseline", "stateping"), ("stateping", "baseline")):
            record, d, c = _play_game(opening, white, black, policies, max_plies=max_plies, board_factory=factory)
            records.append(record)
            disagreements += d
            compared += c
    baseline_wins = stateping_wins = draws = 0
    for record in records:
        if record.result == "1-0":
            winner = record.white
        elif record.result == "0-1":
            winner = record.black
        else:
            winner = None
        if winner == "baseline":
            baseline_wins += 1
        elif winner == "stateping":
            stateping_wins += 1
        else:
            draws += 1
    n = len(records)
    baseline_points = baseline_wins + 0.5 * draws
    state_points = stateping_wins + 0.5 * draws
    return ArenaResult(
        games=tuple(records),
        baseline_wins=baseline_wins,
        stateping_wins=stateping_wins,
        draws=draws,
        baseline_score=baseline_points / n,
        stateping_score=state_points / n,
        mean_game_length=sum(g.plies for g in records) / n,
        move_disagreement_rate=0.0 if compared == 0 else disagreements / compared,
        forfeits=sum(g.forfeit is not None for g in records),
    )


__all__ = ["ArenaGame", "ArenaResult", "play_paired_arena"]
