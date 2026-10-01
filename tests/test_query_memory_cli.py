from chessfly_statepings import cli


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


def test_parser_exposes_query_memory_command():
    args = cli.build_parser().parse_args(["query-memory", "--positions", "p.txt"])
    assert args.command == "query-memory"
    assert args.rho == 0.75
    assert args.seed == 0


def test_query_memory_command_writes_three_arm_receipt(tmp_path, monkeypatch):
    positions = tmp_path / "p.txt"
    positions.write_text("x\ny\n", encoding="utf-8")
    loader, models = _fake_runtime()

    class Metrics:
        accuracy = 0.5
        mean_reciprocal_rank = 0.75
        mean_correct_margin = 0.1

    class Result:
        rho = 0.75
        seed = 7
        positions = 2
        present = Metrics()
        history = Metrics()
        shuffled_history = Metrics()

    monkeypatch.setattr(cli, "evaluate_query_memory", lambda *a, **k: Result())
    captured = {}

    def receipt(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "build_receipt", receipt)
    monkeypatch.setattr(cli, "write_receipt", lambda *a, **k: None)

    rc = cli.main([
        "query-memory", "--positions", str(positions), "--seed", "7",
        "--output", str(tmp_path / "r.json")
    ], artifact_loader=loader, model_loader=models)

    assert rc == 0
    assert captured["command"] == "query-memory"
    assert set(captured["results"]["arms"]) == {"present", "history", "shuffled_history"}
