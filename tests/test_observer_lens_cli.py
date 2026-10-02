from chessfly_statepings import cli


def test_parser_exposes_observer_lens_commands():
    curvature = cli.build_parser().parse_args(
        ["curvature", "--positions", "p.txt"]
    )
    assert curvature.rho == 0.75
    assert curvature.magnitudes == [0.25, 0.5, 1.0, 2.0, 4.0]

    crossing = cli.build_parser().parse_args(
        ["state-crossing", "--positions", "p.txt"]
    )
    assert crossing.controls == 32
    assert crossing.magnitudes == [0.5, 1.0, 2.0]

    feedback = cli.build_parser().parse_args(
        ["feedback-drift", "--positions", "p.txt"]
    )
    assert feedback.magnitudes == [0.5, 1.0, 2.0]
    assert feedback.gammas == [0.0, 0.01, 0.05]
