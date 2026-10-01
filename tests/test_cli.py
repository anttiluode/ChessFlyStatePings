import pytest

from chessfly_statepings import cli


def test_cli_has_all_v0_commands():
    parser = cli.build_parser()
    for argv in (["assets"], ["probe", "--fen", "x"], ["compare", "--positions", "p"], ["sweep", "--positions", "p"], ["arena", "--games", "2"], ["orthogonal", "--positions", "p"]):
        args = parser.parse_args(argv)
        assert args.command == argv[0]


def test_assets_command_uses_injected_artifact_loader(tmp_path):
    calls = []

    class M:
        def to_dict(self): return {"ok": True}

    def loader(cache_dir=None, **kwargs):
        calls.append((cache_dir, kwargs))
        return M()

    rc = cli.main(["assets", "--cache-dir", str(tmp_path)], artifact_loader=loader)
    assert rc == 0
    assert calls and calls[0][0] == str(tmp_path)


def test_default_sweep_retains_all_declared_settings():
    grid = cli.default_sweep()
    assert len(grid) == 28
    assert (0.25, -0.20) in grid
    assert (0.90, 0.20) in grid
    assert all(any(k == 0 for r, k in grid if r == rho) for rho in (0.25, 0.5, 0.75, 0.9))


def test_help_does_not_need_external_artifacts():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def _fake_runtime():
    class Manifest:
        paths = type("P", (), {})()
        def to_dict(self): return {"ok": True}

    class W:
        step = None
        steps = 5
        alpha = 0.5
        hidden = 4
        encoder_inputs = 1
        readout_neurons = 1

    class M:
        device = "cpu"

    def loader(*args, **kwargs): return Manifest()
    def models(manifest, device): return None, W(), M(), M()
    return loader, models


def test_sweep_receipt_accumulates_instability_counts(tmp_path, monkeypatch):
    positions = tmp_path / "p.txt"
    positions.write_text("x\n")
    loader, models = _fake_runtime()

    class R:
        rows = ()
        aggregate = {"instability_count": 1.0}

    monkeypatch.setattr(cli, "compare_positions", lambda *a, **k: R())
    captured = {}

    def receipt(**kwargs):
        captured.update(kwargs)
        return {"x": 1}

    monkeypatch.setattr(cli, "build_receipt", receipt)
    monkeypatch.setattr(cli, "write_receipt", lambda *a, **k: None)
    rc = cli.main(["sweep", "--positions", str(positions), "--output", str(tmp_path / "r.json")], artifact_loader=loader, model_loader=models)
    assert rc == 0
    assert captured["instability_count"] == 28


def test_orthogonal_command_writes_all_declared_runs(tmp_path, monkeypatch):
    positions = tmp_path / "p.txt"
    positions.write_text("x\n")
    loader, models = _fake_runtime()

    class Run:
        lambda_value = 2.0
        rows = ()
        aggregate = {"positions": 1.0}

    class Result:
        rho = 0.75
        seed = 0
        runs = (Run(),)

    monkeypatch.setattr(cli, "evaluate_orthogonal_positions", lambda *a, **k: Result())
    captured = {}

    def receipt(**kwargs):
        captured.update(kwargs)
        return {"x": 1}

    monkeypatch.setattr(cli, "build_receipt", receipt)
    monkeypatch.setattr(cli, "write_receipt", lambda *a, **k: None)
    rc = cli.main(["orthogonal", "--positions", str(positions), "--output", str(tmp_path / "r.json")], artifact_loader=loader, model_loader=models)
    assert rc == 0
    assert captured["command"] == "orthogonal"
    assert captured["results"]["settings"] == 1
    assert captured["results"]["runs"][0]["lambda"] == 2.0
