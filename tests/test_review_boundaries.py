import gzip
import hashlib
import json

import pytest

from chessfly_statepings import assets
from chessfly_statepings.arena import play_paired_arena


def _meta(raw_connectome: bytes, raw_neurons: bytes):
    return {
        "dataset": "fixture",
        "files": {
            "connectome.bin.gz": {"rawSha256": hashlib.sha256(raw_connectome).hexdigest()},
            "neurons.bin.gz": {"rawSha256": hashlib.sha256(raw_neurons).hexdigest()},
        },
    }


def test_model_revision_is_not_invented_from_space_header(tmp_path, monkeypatch):
    raw_connectome = b"connectome"
    raw_neurons = b"neurons"
    meta = _meta(raw_connectome, raw_neurons)
    payloads = {
        "meta.json": json.dumps(meta).encode(),
        "connectome.bin.gz": gzip.compress(raw_connectome),
        "neurons.bin.gz": gzip.compress(raw_neurons),
        "flynet.safetensors": b"weights",
    }

    def fake_fetch(url):
        name = "meta.json" if url.endswith("meta.json") else url.split("/")[-1].split("?")[0]
        headers = {"x-repo-commit": "space-commit"} if name != "flynet.safetensors" else {}
        return payloads[name], headers

    monkeypatch.setattr(assets, "_fetch_bytes", fake_fetch)
    manifest = assets.ensure_artifacts(tmp_path)
    model = next(record for record in manifest.files if record.name == "flynet.safetensors")
    assert model.resolved_revision is None


class Move:
    def __init__(self, uci): self._uci = uci
    def uci(self): return self._uci


class Decision:
    def __init__(self, uci):
        self.move = Move(uci)
        self.selected_uci = uci


class Policy:
    def __init__(self, uci): self.uci = uci
    def select(self, board): return Decision(self.uci)


class BuggyPolicy:
    def select(self, board):
        raise RuntimeError("programming bug")


class Board:
    def __init__(self, _fen): self.turn = True; self.plies = 0
    def is_game_over(self, claim_draw=True): return self.plies >= 10
    def result(self, claim_draw=True): return "1/2-1/2"
    def push(self, move): self.plies += 1; self.turn = not self.turn


def test_unrelated_runtime_error_is_not_disguised_as_forfeit():
    with pytest.raises(RuntimeError, match="programming bug"):
        play_paired_arena(BuggyPolicy(), Policy("b2b3"), openings=["x"], games=2, max_plies=4, board_factory=Board)
