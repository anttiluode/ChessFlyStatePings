"""Paired-position evaluation for baseline versus StatePing policies."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .policy import PolicyInstabilityError


@dataclass(frozen=True, slots=True)
class CompareRow:
    fen: str
    baseline_move: str
    stateping_move: str | None
    move_changed: bool | None
    js_divergence: float | None
    baseline_value: float | None
    stateping_value: float | None
    value_delta: float | None
    baseline_activity: Mapping[str, float]
    stateping_activity: Mapping[str, float]
    engine: Mapping[str, Any] | None = None
    instability: str | None = None


@dataclass(frozen=True, slots=True)
class CompareResult:
    rows: tuple[CompareRow, ...]
    aggregate: Mapping[str, float]


def _default_board_factory(fen: str):
    try:
        import chess
    except ImportError as exc:
        raise RuntimeError("python-chess is required for chess evaluation") from exc
    return chess.Board(fen)


def _js_divergence(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        p, q = float(a.get(key, 0.0)), float(b.get(key, 0.0))
        m = 0.5 * (p + q)
        if p > 0 and m > 0:
            total += 0.5 * p * math.log(p / m)
        if q > 0 and m > 0:
            total += 0.5 * q * math.log(q / m)
    return total


class StockfishScorer:
    """Optional local Stockfish wrapper. Construct only when engine scoring is requested."""

    def __init__(self, path: str | None = None, *, depth: int = 12) -> None:
        try:
            import chess.engine
        except ImportError as exc:
            raise RuntimeError("python-chess with engine support is required") from exc
        resolved = path or shutil.which("stockfish")
        if not resolved:
            raise FileNotFoundError("Stockfish executable not found")
        self._chess_engine = chess.engine
        self.engine = chess.engine.SimpleEngine.popen_uci(resolved)
        self.depth = int(depth)

    def close(self) -> None:
        self.engine.quit()

    def score(self, board: Any, move: Any) -> dict[str, Any]:
        child = board.copy(stack=False)
        child.push(move)
        info = self.engine.analyse(child, self._chess_engine.Limit(depth=self.depth))
        score = info["score"].pov(board.turn)
        return {"score": score.score(mate_score=100000), "mate": score.mate()}


def compare_positions(
    fens: Iterable[str],
    baseline_policy: Any,
    stateping_policy: Any,
    *,
    board_factory: Callable[[str], Any] | None = None,
    stockfish: StockfishScorer | None = None,
) -> CompareResult:
    factory = board_factory or _default_board_factory
    rows: list[CompareRow] = []
    for fen in fens:
        board = factory(fen)
        try:
            baseline = baseline_policy.select(board)
        except PolicyInstabilityError as exc:
            rows.append(CompareRow(fen, None, None, None, None, None, None, None, {}, {}, None, f"baseline: {exc}"))
            continue
        try:
            stateping = stateping_policy.select(board)
        except PolicyInstabilityError as exc:
            rows.append(CompareRow(fen, baseline.selected_uci, None, None, None, baseline.value_expectation, None, None, dict(baseline.activity), {}, None, f"stateping: {exc}"))
            continue
        engine = None
        if stockfish is not None:
            engine = {
                "baseline": stockfish.score(board, baseline.move),
                "stateping": stockfish.score(board, stateping.move),
            }
        rows.append(CompareRow(
            fen=fen,
            baseline_move=baseline.selected_uci,
            stateping_move=stateping.selected_uci,
            move_changed=baseline.selected_uci != stateping.selected_uci,
            js_divergence=_js_divergence(baseline.legal_probabilities, stateping.legal_probabilities),
            baseline_value=baseline.value_expectation,
            stateping_value=stateping.value_expectation,
            value_delta=stateping.value_expectation - baseline.value_expectation,
            baseline_activity=dict(baseline.activity),
            stateping_activity=dict(stateping.activity),
            engine=engine,
            instability=None,
        ))
    n = len(rows)
    valid = [row for row in rows if row.instability is None]
    nv = len(valid)
    aggregate = {
        "positions": float(n),
        "valid_positions": float(nv),
        "instability_count": float(n - nv),
        "move_change_rate": 0.0 if nv == 0 else sum(bool(row.move_changed) for row in valid) / nv,
        "mean_js_divergence": 0.0 if nv == 0 else sum(float(row.js_divergence) for row in valid) / nv,
        "mean_value_delta": 0.0 if nv == 0 else sum(float(row.value_delta) for row in valid) / nv,
    }
    return CompareResult(tuple(rows), aggregate)


__all__ = ["CompareResult", "CompareRow", "StockfishScorer", "compare_positions"]
