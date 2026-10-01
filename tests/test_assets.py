import gzip
import hashlib
import json
from pathlib import Path

import pytest

from chessfly_statepings import assets


def _meta(raw_connectome: bytes, raw_neurons: bytes):
    return {
        "dataset": "fixture",
        "files": {
            "connectome.bin.gz": {"rawSha256": hashlib.sha256(raw_connectome).hexdigest()},
            "neurons.bin.gz": {"rawSha256": hashlib.sha256(raw_neurons).hexdigest()},
        },
    }


def test_ensure_artifacts_downloads_atomically_and_reuses_verified_cache(tmp_path, monkeypatch):
    raw_connectome = b"connectome-fixture"
    raw_neurons = b"neurons-fixture"
    payloads = {
        "meta.json": json.dumps(_meta(raw_connectome, raw_neurons)).encode(),
        "connectome.bin.gz": gzip.compress(raw_connectome),
        "neurons.bin.gz": gzip.compress(raw_neurons),
        "flynet.safetensors": b"weights-fixture",
    }
    calls = []

    def fake_fetch(url):
        calls.append(url)
        name = "meta.json" if url.endswith("meta.json") else url.split("/")[-1].split("?")[0]
        return payloads[name], {"x-repo-commit": "deadbeef"}

    monkeypatch.setattr(assets, "_fetch_bytes", fake_fetch)
    manifest = assets.ensure_artifacts(tmp_path)
    assert manifest.dataset == "fixture"
    assert manifest.paths.connectome.read_bytes() == payloads["connectome.bin.gz"]
    assert not list(tmp_path.glob("*.tmp"))
    first_calls = list(calls)

    def forbidden(_url):
        raise AssertionError("verified cache should not access network")

    monkeypatch.setattr(assets, "_fetch_bytes", forbidden)
    again = assets.ensure_artifacts(tmp_path)
    assert again.files[0].sha256 == manifest.files[0].sha256
    assert calls == first_calls


def test_hash_mismatch_is_fatal_and_bad_file_not_kept(tmp_path, monkeypatch):
    raw_connectome = b"actual"
    raw_neurons = b"neurons"
    meta = _meta(b"different", raw_neurons)
    payloads = {
        "meta.json": json.dumps(meta).encode(),
        "connectome.bin.gz": gzip.compress(raw_connectome),
        "neurons.bin.gz": gzip.compress(raw_neurons),
        "flynet.safetensors": b"weights",
    }

    def fake_fetch(url):
        name = "meta.json" if url.endswith("meta.json") else url.split("/")[-1].split("?")[0]
        return payloads[name], {}

    monkeypatch.setattr(assets, "_fetch_bytes", fake_fetch)
    with pytest.raises(assets.ArtifactError, match="decoded SHA-256 mismatch"):
        assets.ensure_artifacts(tmp_path)
    assert not (tmp_path / "connectome.bin.gz").exists()


def test_default_cache_is_user_local(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "user_cache_path", lambda name: tmp_path / name)
    assert assets.default_cache_dir() == tmp_path / "chessfly-statepings" / "artifacts"


def test_gitignore_excludes_large_upstream_artifacts():
    text = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")
    for token in ("flynet.safetensors", "connectome.bin*", "neurons.bin*", ".cache/", "artifacts/"):
        assert token in text
