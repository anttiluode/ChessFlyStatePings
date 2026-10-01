"""ChessFly board encoding and action-space utilities."""

from __future__ import annotations

FEATURES = 780
POLICY_ACTIONS = 1968
_FILES = "abcdefgh"
_RANKS = "12345678"


class EncodingError(ValueError):
    pass


def _toggle_case(text: str) -> str:
    return "".join(ch.swapcase() if ch.isalpha() else ch for ch in text)


def mirror_fen(fen: str) -> str:
    fields = fen.split()
    if len(fields) < 4:
        raise EncodingError("FEN must contain at least four fields")
    placement, turn, castling, ep, *rest = fields
    if turn not in {"w", "b"}:
        raise EncodingError("invalid active color")
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise EncodingError("FEN must contain eight ranks")
    mirrored_castling = "-" if castling == "-" else "".join(ch for ch in "KQkq" if ch in _toggle_case(castling))
    if ep == "-":
        mirrored_ep = "-"
    else:
        if len(ep) != 2 or ep[0] not in _FILES or ep[1] not in _RANKS:
            raise EncodingError("invalid en-passant square")
        mirrored_ep = ep[0] + str(9 - int(ep[1]))
    return " ".join([
        "/".join(_toggle_case(rank) for rank in reversed(ranks)),
        "b" if turn == "w" else "w",
        mirrored_castling,
        mirrored_ep,
        *rest,
    ])


def canonical_fen(fen: str) -> tuple[str, bool]:
    fields = fen.split()
    if len(fields) < 2 or fields[1] not in {"w", "b"}:
        raise EncodingError("invalid FEN active color")
    return (mirror_fen(fen), True) if fields[1] == "b" else (fen, False)


def mirror_uci(uci: str) -> str:
    if len(uci) < 4 or uci[0] not in _FILES or uci[2] not in _FILES:
        raise EncodingError(f"invalid UCI move {uci!r}")
    try:
        return f"{uci[0]}{9-int(uci[1])}{uci[2]}{9-int(uci[3])}{uci[4:]}"
    except (ValueError, IndexError) as exc:
        raise EncodingError(f"invalid UCI move {uci!r}") from exc


def _square(rank: int, file_index: int) -> str:
    return _FILES[file_index] + _RANKS[rank]


def build_action_space() -> tuple[str, ...]:
    actions: set[str] = set()
    rays = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
    knights = ((-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1))
    for rank in range(8):
        for file_index in range(8):
            for dr, df in rays:
                rr, ff = rank + dr, file_index + df
                while 0 <= rr < 8 and 0 <= ff < 8:
                    actions.add(_square(rank, file_index) + _square(rr, ff))
                    rr += dr; ff += df
            for dr, df in knights:
                rr, ff = rank + dr, file_index + df
                if 0 <= rr < 8 and 0 <= ff < 8:
                    actions.add(_square(rank, file_index) + _square(rr, ff))
    for rank in (1, 6):
        target_rank = 0 if rank == 1 else 7
        for file_index in range(8):
            for df in (-1, 0, 1):
                ff = file_index + df
                if 0 <= ff < 8:
                    for promotion in "qrbn":
                        actions.add(_square(rank, file_index) + _square(target_rank, ff) + promotion)
    result = tuple(sorted(actions))
    if len(result) != POLICY_ACTIONS:
        raise RuntimeError(f"action-space construction produced {len(result)}, expected {POLICY_ACTIONS}")
    return result


ACTION_SPACE = build_action_space()
ACTION_INDEX = {move: i for i, move in enumerate(ACTION_SPACE)}


def encode_fen(fen: str) -> tuple[float, ...]:
    fields = fen.split()
    if len(fields) < 4:
        raise EncodingError("FEN must contain at least four fields")
    placement, turn, castling, ep = fields[:4]
    if turn != "w":
        raise EncodingError("encode_fen expects canonical white-to-move FEN")
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise EncodingError("FEN must contain eight ranks")
    values = [0.0] * FEATURES
    channels = {"p":0,"n":1,"b":2,"r":3,"q":4,"k":5}
    for fen_rank, rank_text in enumerate(ranks):
        file_index = 0
        for ch in rank_text:
            if ch.isdigit():
                file_index += int(ch)
                continue
            channel = channels.get(ch.lower())
            if channel is None or file_index >= 8:
                raise EncodingError("invalid FEN placement")
            color_offset = 0 if ch.isupper() else 6
            square = (7 - fen_rank) * 8 + file_index
            values[square * 12 + color_offset + channel] = 1.0
            file_index += 1
        if file_index != 8:
            raise EncodingError("invalid FEN rank width")
    if castling != "-":
        for i, flag in enumerate("KQkq"):
            values[768 + i] = 1.0 if flag in castling else 0.0
    if ep != "-":
        if len(ep) != 2 or ep[0] not in _FILES or ep[1] not in _RANKS:
            raise EncodingError("invalid en-passant square")
        values[772 + _FILES.index(ep[0])] = 1.0
    return tuple(values)


__all__ = ["ACTION_INDEX", "ACTION_SPACE", "EncodingError", "canonical_fen", "encode_fen", "mirror_fen", "mirror_uci"]
