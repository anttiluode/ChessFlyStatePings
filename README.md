# ChessFlyStatePings

Experimental research harness for asking whether a small history-dependent component of artificial unit activity changes useful computation in the published **ChessFly** model while keeping its trained checkpoint and fly-derived graph fixed.

This repository does **not** contain the ChessFly checkpoint or connectome files. On first real run it downloads them from the original Hugging Face repositories into a user-local cache and records hashes/provenance. See `ATTRIBUTION.md` and the design/spec under `docs/superpowers/`.

Scientific boundary: ChessFly uses artificial recurrent units wired by a fly-derived graph. `StatePing` is a computational intervention, not a claim that Drosophila neurons use the same code.
