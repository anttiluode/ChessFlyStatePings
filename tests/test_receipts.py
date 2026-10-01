import json

from chessfly_statepings.receipts import build_receipt, write_receipt


def test_receipt_contains_reproducibility_essentials(tmp_path):
    manifest={"space_revision":"abc","model_revision":"def","files":[{"name":"w","sha256":"123"}]}
    payload=build_receipt(command="compare", arguments={"rho":0.75,"kappa":0.1}, device="cpu", artifact_manifest=manifest, model_metadata={"steps":5,"alpha":0.5}, inputs=["fen"], results={"n":1}, instability_count=0)
    for key in ("timestamp_utc","command","arguments","versions","device","artifacts","model","inputs","results","instability_count"):
        assert key in payload
    path=tmp_path/"receipt.json"
    write_receipt(path,payload)
    assert json.loads(path.read_text())["arguments"]["kappa"] == 0.1
