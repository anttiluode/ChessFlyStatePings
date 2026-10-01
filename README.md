# ChessFlyStatePings

`ChessFlyStatePings` is an experimental research harness around Maxime Labonne's published **ChessFly** model. It asks one narrow question inspired by `BrainAsInverseModelerV3`:

> Keep the trained ChessFly checkpoint and fly-derived graph fixed. What happens if recurrent messages can carry a small fast-minus-slow component of each artificial unit's recent settling trajectory?

This is **not** a claim that ChessFly is a biophysical fly brain or that Drosophila spikes use this code. ChessFly's units are artificial recurrent units wired by a fly-derived graph.

## External artifacts stay external

The repository never commits `flynet.safetensors`, `connectome.bin.gz`, or `neurons.bin.gz`. On first real run they are downloaded directly from the original Hugging Face locations and cached under the platform's user cache directory (`chessfly-statepings/artifacts`). The graph files are checked against decoded hashes published by the ChessFly Space metadata; all files get local SHA-256 records in `manifest.json`.

See `ATTRIBUTION.md` for provenance and licensing boundaries.

## Install

```bash
python -m pip install -e .
```

The runtime needs PyTorch, NumPy, safetensors, platformdirs, and python-chess. CUDA is optional; `--device auto` uses CUDA only when PyTorch reports it available, otherwise CPU.

## First run

You can explicitly acquire the artifacts:

```bash
chessfly-statepings assets
```

but `probe`, `compare`, `sweep`, and `arena` call the same verified acquisition automatically.

Observe the **unaltered** ChessFly settling trajectory:

```bash
chessfly-statepings probe --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --rho 0.75
```

Compare original versus StatePing on the same positions:

```bash
chessfly-statepings compare --positions data/smoke_fens.txt --rho 0.75 --kappa 0.10
```

Run the full declared exploratory grid (`rho` 0.25/0.50/0.75/0.90 × `kappa` -0.20…+0.20):

```bash
chessfly-statepings sweep --positions data/smoke_fens.txt
```

Run paired-color headless games using one raw network forward per move:

```bash
chessfly-statepings arena --games 20 --rho 0.75 --kappa 0.10
```

Every nontrivial experiment writes a small JSON receipt containing configuration, versions, upstream artifact hashes/revisions, inputs, model metadata, results, and instability counts. Use `--output path.json` to choose its location.

## The intervention

For ordinary ChessFly activity `h_t`, define a local slow state and residue during the K-step settle:

```text
s_(t+1) = rho s_t + (1-rho) h_(t+1)
r_(t+1) = h_(t+1) - s_(t+1)
```

The experimental recurrent message is

```text
q_t       = W h_t
q_res_t   = W r_t
q'_t      = q_t + kappa q_res_t
```

Everything else stays frozen. `kappa=0` is the exact control path. v0 does **not** carry neural state between actual chess moves and does not retrain any of the roughly 15M learned edge gains.

## Measurement order

1. `probe`: observe whether fast/slow structure exists without changing the model.
2. `compare`: paired positions measure move changes, policy Jensen-Shannon divergence, value shifts, and activity/residue statistics.
3. `sweep`: retain every declared setting rather than cherry-picking the best one.
4. `arena`: descriptive paired-color games. Match wins alone are not an Elo estimate or an improvement claim.

Stockfish is optional and intended for stricter move-quality checks; the baseline-vs-StatePing comparison and arena do not require it.

## Tests

```bash
python -m pip install -e '.[test]'
pytest -q
```

The default suite uses tiny synthetic graphs and does not download the real 150MB-class artifacts. To run the opt-in real-artifact compatibility check:

```bash
CHESSFLY_RUN_INTEGRATION=1 pytest tests/test_integration_real.py -q
```

## Scientific boundary

A positive result would show that the frozen ChessFly computation is sensitive to a particular history-bearing coordinate. It would not establish a biological waveform code, consciousness, or that a fly connectome is intrinsically suited to chess. A negative or destabilizing result is equally valid evidence for this intervention under the frozen checkpoint.
