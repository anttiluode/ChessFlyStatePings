# ChessFlyStatePings

`ChessFlyStatePings` is an experimental research harness around Maxime Labonne's published **ChessFly** model. It asks narrow questions inspired by `BrainAsInverseModelerV3` while keeping the trained checkpoint and fly-derived graph fixed.

The first StatePing gate tested whether recurrent messages could carry a fast-minus-slow component of each artificial unit's recent settling trajectory. The first real-artifact run showed that this residue was almost perfectly collinear with current activity, so the next gate explicitly removes that present-state direction before asking whether any structured history remains readable.

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

but all experiment commands call the same verified acquisition automatically.

Observe the **unaltered** ChessFly settling trajectory:

```bash
chessfly-statepings probe --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --rho 0.75
```

Compare original versus recurrent StatePing on the same positions:

```bash
chessfly-statepings compare --positions data/smoke_fens.txt --rho 0.75 --kappa 0.10
```

Run the full declared recurrent sweep (`rho` 0.25/0.50/0.75/0.90 × `kappa` -0.20…+0.20):

```bash
chessfly-statepings sweep --positions data/smoke_fens.txt
```

Run the orthogonal-history readout gate:

```bash
chessfly-statepings orthogonal --positions data/smoke_fens.txt --rho 0.75 --seed 0
```

Run paired-color headless games using one raw network forward per move:

```bash
chessfly-statepings arena --games 20 --rho 0.75 --kappa 0.10
```

Every nontrivial experiment writes a small JSON receipt containing configuration, versions, upstream artifact hashes/revisions, inputs, model metadata, results, and instability counts. Use `--output path.json` to choose its location.

## Gate 0: recurrent fast-minus-slow StatePing

For ordinary ChessFly activity `h_t`, define a local slow state and residue during the K-step settle:

```text
s_(t+1) = rho s_t + (1-rho) h_(t+1)
r_(t+1) = h_(t+1) - s_(t+1)
```

The first experimental recurrent message is

```text
q_t       = W h_t
q_res_t   = W r_t
q'_t      = q_t + kappa q_res_t
```

Everything else stays frozen. `kappa=0` is the exact control path. This gate does **not** carry neural state between actual chess moves and does not retrain any of the roughly 15M learned edge gains.

The first real run found that `r_t` was almost parallel to `h_t` in the readout population, so this intervention mostly behaved like a recurrent-gain change rather than an independent history coordinate.

## Gate 1: orthogonal history readout

The next gate leaves ChessFly's recurrence completely unchanged. At the final readout population it forms the same fast-minus-slow residue, then removes the component parallel to the current state:

```text
r_perp = r - ((r · h) / (h · h + eps)) h
```

The frozen ChessFly decoder is then evaluated on

```text
h_readout + lambda r_perp
```

and compared with a matched shuffled control

```text
h_readout + lambda shuffle(r_perp)
```

The declared `lambda` sweep is `-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4`. The same deterministic shuffle is reused across all lambdas for a position.

Each run records:

- `mean_orthogonal_energy_ratio`: how much residue norm survives after removing the present-state direction;
- `mean_abs_orthogonal_cosine`: a numerical correctness check that should be near zero;
- real versus shuffled move-change rates;
- real versus shuffled legal-policy Jensen-Shannon divergence;
- real versus shuffled absolute value shifts;
- `real_minus_shuffled_js`, where positive means the real orthogonal direction perturbed the frozen decoder more than its shuffled control. This is **not** by itself a chess-strength score.

Interpretation is deliberately narrow:

- orthogonal energy near zero means this EMA history coordinate contains almost no independent readout direction;
- substantial energy but real ≈ shuffled means an independent direction exists but the frozen decoder is not specially aligned to its structure;
- a reproducible real-versus-shuffled difference means structured trajectory geometry survives beyond simple present-state amplitude and is readable by the frozen decoder.

## Measurement order

1. `probe`: observe unaltered settling dynamics.
2. `compare`: measure the original recurrent StatePing intervention.
3. `sweep`: retain the entire declared recurrent grid rather than cherry-picking.
4. `orthogonal`: remove the present-state direction and compare real trajectory structure against a shuffled matched control.
5. `arena`: descriptive paired-color games. Match wins alone are not an Elo estimate or an improvement claim.

Stockfish is optional and intended for stricter move-quality checks; the baseline-vs-StatePing comparison and arena do not require it.

## Tests

```bash
python -m pip install -e '.[test]'
pytest -q
```

The default suite uses tiny synthetic graphs and does not download the real 150MB-class artifacts. To run the opt-in real-artifact compatibility check on macOS/Linux shells:

```bash
CHESSFLY_RUN_INTEGRATION=1 pytest tests/test_integration_real.py -q
```

On Windows `cmd.exe` use:

```bat
set CHESSFLY_RUN_INTEGRATION=1 && python -m pytest tests\test_integration_real.py -q
```

## Scientific boundary

A positive result would show that the frozen ChessFly computation is sensitive to a particular history-bearing coordinate. It would not establish a biological waveform code, consciousness, or that a fly connectome is intrinsically suited to chess. A negative or destabilizing result is equally valid evidence for the tested coordinate under the frozen checkpoint.
