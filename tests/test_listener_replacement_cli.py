from chessfly_statepings import cli


def _fake_runtime():
    class Manifest:
        paths = type("P", (), {})()

        def to_dict(self):
            return {"ok": True}

    class W:
        step = None
        steps = 5
        alpha = 0.5
        hidden = 4
        encoder_inputs = 1
        readout_neurons = 1

    class M:
        device = "cpu"

    def loader(*args, **kwargs):
        return Manifest()

    def models(manifest, device):
        return None, W(), M(), M()

    return loader, models


def test_parser_exposes_listener_replacement_command():
    args = cli.build_parser().parse_args(
        ["listener-replacement", "--positions", "p.txt"]
    )
    assert args.command == "listener-replacement"
    assert args.rho == 0.75
    assert args.seed == 0
    assert args.controls == 32
    assert args.magnitudes == [1.0, 2.0, 4.0]


def test_listener_replacement_command_writes_listener_receipt(tmp_path, monkeypatch):
    positions = tmp_path / "p.txt"
    positions.write_text("x\ny\n", encoding="utf-8")
    loader, models = _fake_runtime()

    class Metrics:
        accuracy = 0.5
        mean_reciprocal_rank = 0.75
        mean_correct_margin = 0.1

    class Details:
        scores = ((1.0, 0.2), (0.3, 1.0))
        ranks = (1.0, 1.0)
        margins = (0.8, 0.7)

    class Run:
        magnitude = 2.0
        association = Metrics()
        association_details = Details()
        value = Metrics()
        policy = Metrics()
        value_details = Details()
        policy_details = Details()
        value_centered = Metrics()
        policy_centered = Metrics()
        value_centered_details = Details()
        policy_centered_details = Details()
        value_control_mean = Metrics()
        policy_control_mean = Metrics()
        value_percentiles = {
            "accuracy": 0.8,
            "mean_reciprocal_rank": 0.9,
            "mean_correct_margin": 0.95,
        }
        policy_percentiles = {
            "accuracy": 0.6,
            "mean_reciprocal_rank": 0.7,
            "mean_correct_margin": 0.75,
        }
        value_centered_control_mean = Metrics()
        policy_centered_control_mean = Metrics()
        value_centered_percentiles = {
            "accuracy": 0.9,
            "mean_reciprocal_rank": 0.95,
            "mean_correct_margin": 0.97,
        }
        policy_centered_percentiles = {
            "accuracy": 0.55,
            "mean_reciprocal_rank": 0.65,
            "mean_correct_margin": 0.7,
        }

    class Result:
        rho = 0.75
        seed = 7
        positions = 2
        controls = 32
        raw_history = Metrics()
        raw_history_details = Details()
        runs = (Run(),)

    monkeypatch.setattr(cli, "evaluate_listener_replacement", lambda *a, **k: Result())
    captured = {}

    def receipt(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "build_receipt", receipt)
    monkeypatch.setattr(cli, "write_receipt", lambda *a, **k: None)

    rc = cli.main(
        [
            "listener-replacement",
            "--positions",
            str(positions),
            "--seed",
            "7",
            "--controls",
            "32",
            "--magnitudes",
            "2",
            "--output",
            str(tmp_path / "r.json"),
        ],
        artifact_loader=loader,
        model_loader=models,
    )

    assert rc == 0
    assert captured["command"] == "listener-replacement"
    payload = captured["results"]
    assert set(payload) == {
        "rho",
        "seed",
        "positions",
        "controls",
        "raw_history",
        "raw_history_details",
        "runs",
    }
    run = payload["runs"][0]
    assert run["magnitude"] == 2.0
    assert set(run) == {
        "magnitude",
        "association",
        "association_details",
        "value",
        "policy",
    }
    assert set(run["value"]) == {
        "real",
        "control_mean",
        "percentiles",
        "details",
        "centered",
    }
    assert set(run["value"]["centered"]) == {
        "real",
        "control_mean",
        "percentiles",
        "details",
    }
    assert run["value"]["centered"]["details"]["ranks"] == [1.0, 1.0]
