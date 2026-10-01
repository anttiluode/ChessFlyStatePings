import gzip
import hashlib
import json

from chessfly_statepings import assets
from chessfly_statepings.receipts import build_receipt


def _meta(raw_connectome: bytes, raw_neurons: bytes):
    return {
        "dataset": "fixture",
        "files": {
            "connectome.bin.gz": {"rawSha256": hashlib.sha256(raw_connectome).hexdigest()},
            "neurons.bin.gz": {"rawSha256": hashlib.sha256(raw_neurons).hexdigest()},
        },
    }


def test_requested_revision_change_refreshes_cache(tmp_path, monkeypatch):
    raw_connectome = b"connectome"
    raw_neurons = b"neurons"
    payloads = {
        "meta.json": json.dumps(_meta(raw_connectome, raw_neurons)).encode(),
        "connectome.bin.gz": gzip.compress(raw_connectome),
        "neurons.bin.gz": gzip.compress(raw_neurons),
        "flynet.safetensors": b"weights",
    }
    calls = []

    def fake_fetch(url):
        calls.append(url)
        name = "meta.json" if url.endswith("meta.json") else url.split("/")[-1].split("?")[0]
        return payloads[name], {"x-repo-commit": "resolved"}

    monkeypatch.setattr(assets, "_fetch_bytes", fake_fetch)
    assets.ensure_artifacts(tmp_path, space_revision="space-a", model_revision="model-a")
    first_count = len(calls)
    manifest = assets.ensure_artifacts(tmp_path, space_revision="space-b", model_revision="model-b")
    assert len(calls) > first_count
    assert manifest.space_revision == "space-b"
    assert manifest.model_revision == "model-b"


def test_receipt_records_torch_thread_configuration():
    payload = build_receipt(
        command="compare",
        arguments={"rho": 0.75, "kappa": 0.1},
        device="cpu",
        artifact_manifest={"files": []},
        model_metadata={"steps": 5},
        inputs=["fen"],
        results={},
        instability_count=0,
    )
    assert isinstance(payload["torch_threads"], int)
    assert payload["torch_threads"] >= 1
