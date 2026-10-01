from chessfly_statepings.encoding import ACTION_SPACE, canonical_fen, encode_fen, mirror_uci


def test_action_space_is_deterministic_and_has_1968_actions():
    assert len(ACTION_SPACE) == 1968
    assert len(set(ACTION_SPACE)) == 1968
    assert ACTION_SPACE == tuple(sorted(ACTION_SPACE))


def test_black_fen_is_mirrored_to_white_perspective():
    fen = "8/8/8/8/8/8/p7/K6k b - - 0 1"
    canonical, mirrored = canonical_fen(fen)
    assert mirrored is True
    assert canonical.split()[1] == "w"
    assert canonical.split()[0] == "k6K/P7/8/8/8/8/8/8"


def test_mirror_uci_preserves_promotion():
    assert mirror_uci("a2a1q") == "a7a8q"


def test_encode_fen_has_780_features_and_metadata_flags():
    fen = "4k3/8/8/8/8/8/4P3/4K3 w Kq e6 0 1"
    x = encode_fen(fen)
    assert len(x) == 780
    assert sum(x[:768]) == 3.0
    assert x[768] == 1.0
    assert x[771] == 1.0
    assert x[772 + 4] == 1.0
