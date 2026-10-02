import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chessfly_statepings import cli
from test_history_lens import memory_model


def local_cases(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data" / "history_lens_cases.json"
    rows = json.loads(source.read_text())["cases"][:2]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": rows}))
    return path, rows


def loaders():
    model = memory_model()
    manifest = SimpleNamespace(to_dict=lambda: {"fixture": "local-tiny-model"})
    return (lambda *args, **kwargs: manifest), (lambda *args: (model.graph, model.weights, model, model))


def test_history_lens_cli_writes_complete_finite_receipt(tmp_path):
    cases, rows = local_cases(tmp_path)
    output = tmp_path / "history.json"
    artifacts, models = loaders()
    assert cli.main([
        "history-lens", "--cases", str(cases), "--controls", "2", "--seed", "5",
        "--ping-steps", "2", "--delays", "1", "2", "--magnitudes", "0.5", "1", "--output", str(output),
    ], artifact_loader=artifacts, model_loader=models) == 0

    receipt = json.loads(output.read_text())
    assert receipt["schema"] == "chessfly-statepings-receipt-v1"
    assert receipt["command"] == "history-lens"
    assert receipt["inputs"] == rows
    assert receipt["artifacts"]["fixture"] == "local-tiny-model"
    assert receipt["instability_count"] == 0
    result = receipt["results"]
    assert result["seed"] == 5 and result["controls"] == 2
    assert result["case_count"] == 2
    assert result["primary"]["status"] == "inconclusive"
    assert [(r["delay"], r["magnitude"]) for r in result["runs"]] == [(1, 0.5), (1, 1.0), (2, 0.5), (2, 1.0)]
    assert len(result["runs"][0]["cases"][0]["policy"]["controls"]) == 2
    json.dumps(receipt, allow_nan=False)


def test_default_cli_uses_causal_three_step_source_and_held_out_step_five(tmp_path):
    cases, _ = local_cases(tmp_path)
    output = tmp_path / "causal.json"
    artifacts, models = loaders()
    cli.main([
        "history-lens", "--cases", str(cases), "--controls", "1", "--output", str(output),
    ], artifact_loader=artifacts, model_loader=models)
    result = json.loads(output.read_text())["results"]
    assert result["protocol"] == "same-present-history-lens-v2"
    assert result["ping_steps"] == 3
    assert result["primary"]["delay"] == 2
    assert len(result["runs"]) == 3
    for run in result["runs"]:
        assert run["delay"] == 2
        for row in run["cases"]:
            assert row["query_step"] == 4 and row["target_step"] == 5


def test_invalid_history_cases_fail_before_acquiring_artifacts(tmp_path):
    cases = tmp_path / "empty.json"
    cases.write_text('{"cases": []}')
    output = tmp_path / "invalid.json"

    def unexpected_download(*args, **kwargs):
        raise AssertionError("invalid case data must not trigger acquisition")

    with pytest.raises(ValueError, match="nonempty"):
        cli.main(["history-lens", "--cases", str(cases), "--output", str(output)], artifact_loader=unexpected_download)
    assert not output.exists()


def test_omitted_primary_setting_is_not_selected_from_secondary_runs(tmp_path):
    cases, _ = local_cases(tmp_path)
    output = tmp_path / "secondary.json"
    artifacts, models = loaders()
    cli.main([
        "history-lens", "--cases", str(cases), "--controls", "1", "--delays", "2",
        "--magnitudes", "0.5", "--output", str(output),
    ], artifact_loader=artifacts, model_loader=models)
    assert json.loads(output.read_text())["results"]["primary"]["status"] == "not_run"


def test_invalid_delay_emits_no_success_receipt(tmp_path):
    cases, _ = local_cases(tmp_path)
    output = tmp_path / "invalid.json"
    artifacts, models = loaders()
    with pytest.raises(ValueError, match="independent next"):
        cli.main([
            "history-lens", "--cases", str(cases), "--delays", "3", "--output", str(output),
        ], artifact_loader=artifacts, model_loader=models)
    assert not output.exists()
