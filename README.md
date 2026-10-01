# ChessFlyStatePings

`ChessFlyStatePings` is an experimental research harness around Maxime Labonne's published **ChessFly** model. It asks narrow questions inspired by `BrainAsInverseModelerV3` while keeping the trained checkpoint and fly-derived graph fixed.

The first StatePing gate tested whether recurrent messages could carry a fast-minus-slow component of each artificial unit's recent settling trajectory. The first real-artifact run showed that this residue was almost perfectly collinear with current activity, so the next gates explicitly remove that present-state direction and ask whether the surviving trajectory direction is unusually readable.

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

Run the one-control orthogonal-history smoke gate:

```bash
chessfly-statepings orthogonal --positions data/smoke_fens.txt --rho 0.75 --seed 0
```

Run the directional-specificity gate against 32 matched controls, testing both signs at each magnitude:

```bash
chessfly-statepings specificity --positions data/smoke_fens.txt --rho 0.75 --controls 32 --seed 0 --magnitudes 1 2 4
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

This gate leaves ChessFly's recurrence completely unchanged. At the final readout population it forms the same fast-minus-slow residue, then removes the component parallel to the current state:

```text
r_perp = r - ((r · h) / (h · h + eps)) h
```

The frozen ChessFly decoder is then evaluated on

```text
h_readout + lambda r_perp
```

The matched control first permutes the entries of `r_perp`, projects the permutation back into the subspace orthogonal to `h_readout`, and rescales it to the original `||r_perp||`. Thus the real and shuffled conditions have the same perturbation norm and neither can sneak the present-state/amplitude direction back in:

```text
u       = shuffle(r_perp)
u_perp  = u - ((u · h) / (h · h + eps)) h
control = ||r_perp|| * u_perp / (||u_perp|| + eps)
```

The original receipt field `mean_orthogonal_energy_ratio` was misnamed: it stores `||r_perp|| / ||r||`, which is a **norm ratio**. New receipts also report `mean_orthogonal_norm_ratio` and the true squared-energy fraction `mean_orthogonal_energy_fraction = mean((||r_perp|| / ||r||)^2)`. The legacy field remains for compatibility.

The first three-position smoke run found a mean orthogonal norm ratio of about 0.0303, yet some directions caused large frozen-decoder changes. It also exposed a sign confound: the real direction was potent for positive lambda on one position, while one matched shuffled direction was potent for negative lambda. That result motivates Gate 1b rather than a claim that the real trajectory direction is already privileged.

## Gate 1b: directional specificity

Gate 1b asks a stricter question: is the real orthogonal-history direction unusually readable compared with **many** equally sized matched directions?

For each position it generates `N` shuffled, re-orthogonalized, norm-matched controls. For each magnitude `a`, it evaluates both signs:

```text
h + a v
h - a v
```

and defines sign-symmetric sensitivity as the larger effect of the two signs. This prevents choosing the favorable sign after looking at the result.

The default run uses 32 controls and magnitudes `1, 2, 4`. The expensive five-step ChessFly recurrence still runs only once per position. Real/control directions and both signs are batched through the frozen decoder.

Gate 1b records pre-softmax and post-softmax effects:

- centered legal-policy logit RMS change, invariant to a common logit shift;
- absolute change in the legal top-1 versus top-2 logit margin;
- centered value-logit RMS change and winning value bin;
- relative change in the 512-dimensional decoder association vector;
- legal-policy Jensen-Shannon divergence;
- the real direction's empirical percentile among matched controls for every metric.

A real percentile near 0.5 means the trajectory direction is ordinary for that metric. Repeated high percentiles across many held-out positions would be evidence that the real temporal direction is aligned with something the frozen decoder reads unusually strongly. It is still not a chess-strength score.

## Measurement order

1. `probe`: observe unaltered settling dynamics.
2. `compare`: measure the original recurrent StatePing intervention.
3. `sweep`: retain the entire declared recurrent grid rather than cherry-picking.
4. `orthogonal`: remove the present-state direction and compare one real trajectory direction against one matched control.
5. `specificity`: compare the real direction with many matched controls using both signs and pre-softmax metrics.
6. `arena`: descriptive paired-color games. Match wins alone are not an Elo estimate or an improvement claim.

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

A positive result would show that the frozen ChessFly computation is unusually sensitive to a particular history-bearing direction under the tested readout geometry. It would not establish a biological waveform code, consciousness, or that a fly connectome is intrinsically suited to chess. A negative result is equally valid evidence for the tested coordinate under the frozen checkpoint.
