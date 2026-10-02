# ChessFlyStatePings

`ChessFlyStatePings` is an experimental research harness around Maxime Labonne's published **ChessFly** model. It asks narrow questions inspired by `BrainAsInverseModelerV3` while keeping the trained checkpoint and fly-derived graph fixed.

The first StatePing gate tested whether recurrent messages could carry a fast-minus-slow component of each artificial unit's recent settling trajectory. The first real-artifact run showed that this residue was almost perfectly collinear with current activity, so the next gates explicitly remove that present-state direction and ask whether the surviving trajectory direction is unusually readable. Gate 1b found suggestive reader-specific sensitivity in a three-position smoke test; Gate 2 made the stronger "ping as query" metaphor literal and failed; Gate 3 then asked whether the **learned receiver geometry**, rather than the raw history vector, makes that temporal direction more address-like.

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

Run the explicit history-query retrieval gate:

```bash
chessfly-statepings query-memory --positions data/smoke_fens.txt --rho 0.75 --seed 0
```

Run the learned receiver-geometry query gate:

```bash
chessfly-statepings receiver-query --positions data/smoke_fens.txt --rho 0.75 --seed 0 --controls 32 --magnitudes 1 2 4
```

Run paired-color headless games using one raw network forward per move:

```bash
chessfly-statepings arena --games 20 --rho 0.75 --kappa 0.10
```

Every nontrivial experiment writes a small JSON receipt containing configuration, versions, upstream artifact hashes/revisions, inputs, model metadata, results, and instability counts. Use `--output path.json` to choose its location.

Real-run receipts and the current interpretation are kept under [`results/`](results/README.md).

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

A real percentile near 0.5 means the trajectory direction is ordinary for that metric. The three-position CUDA smoke rerun placed the real direction around the 94th percentile for value-logit sensitivity and the 86th percentile at the shared association layer, but only around the 68th percentile for policy-logit sensitivity. That is preliminary evidence of receiver-specific readability, not a chess-strength result.

## Gate 2: explicit query-memory retrieval

Gate 2 tests a stronger claim: does the final state-bearing ping use its history coordinate as an address back into recorded settling history?

For each position, the bank stores the penultimate settling state and its orthogonal history coordinate. The final state is then used as a query. Three fixed, untrained retrieval arms are compared:

- present state only;
- present plus the real orthogonal-history coordinate;
- present plus a shuffled history coordinate from another position.

The score is an equal-weight average of cosine similarities for present and history coordinates. Top-1 retrieval, mean reciprocal rank, and the correct-vs-best-distractor margin are recorded.

The first three-position CUDA smoke test is a **negative result for this literal construction**. Present-only retrieval scored 1.0 top-1 accuracy and 1.0 MRR with a positive mean margin of +0.00676. Adding the real history coordinate reduced accuracy to about 0.333 and MRR to about 0.611, with a negative mean margin of -0.00612. The shuffled-history control remained at 1.0 accuracy/MRR with a +0.00539 margin.

So the tested orthogonal history coordinate is not, by itself, a useful equal-weight cosine key to its own penultimate settling state. This narrows the interpretation of Gate 1b: **history being unusually readable by a trained downstream geometry is not the same claim as history being an explicit retrieval address under an imposed metric.**

## Gate 3: receiver-query geometry

Gate 3 tests the next narrower possibility: the raw history vector may not be an address, but its **effect through an already-trained receiver** may preserve trajectory identity.

For a readout state `h`, history direction `r_perp`, magnitude `a`, and frozen decoder association map `A`, define

```text
g(a) = (A(h + a r_perp) - A(h - a r_perp)) / (2a)
```

This is a sign-symmetric central-difference approximation to the local receiver response along the history direction. The same construction is propagated through the frozen value head. Step 4 produces memory signatures; step 5 produces query signatures. Thirty-two matched controls use the same shuffled feature permutation at both steps, followed by re-orthogonalization and norm matching.

The first three-position CUDA smoke run remains **negative for absolute retrieval**: raw history retrieves 0/3 positions, the present association receiver retrieves 1/3, and the real receiver signatures also retrieve only 1/3 at all magnitudes. All real correct-vs-best-distractor margins remain below zero.

However, the learned **value-head geometry** makes the real temporal signature substantially more predecessor-specific than matched directions. At magnitudes 1, 2, and 4, the real value-space margin percentiles are 86.4% at all three settings and the MRR percentiles are 90.9%, 87.9%, and 89.4%. The real value margins approach zero (-0.00589, -0.00408, -0.00163) while matched-control means remain near -0.20.

This is suggestive receiver-specific enrichment, not successful recall. Association-space enrichment is weaker and declines with magnitude. The smoke result therefore supports a relational hypothesis—history may become more query-like **with respect to a particular learned receiver geometry**—while explicitly stopping short of claiming that Gate 3 has produced a working memory address.

## Measurement order

1. `probe`: observe unaltered settling dynamics.
2. `compare`: measure the original recurrent StatePing intervention.
3. `sweep`: retain the entire declared recurrent grid rather than cherry-picking.
4. `orthogonal`: remove the present-state direction and compare one real trajectory direction against one matched control.
5. `specificity`: compare the real direction with many matched controls using both signs and pre-softmax metrics.
6. `query-memory`: test the stronger literal claim that the history-bearing ping retrieves its own recorded settling predecessor.
7. `receiver-query`: ask whether the frozen learned receiver makes the real temporal direction more predecessor-specific than matched directions.
8. `arena`: descriptive paired-color games. Match wins alone are not an Elo estimate or an improvement claim.

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

The current evidence separates three claims. Gate 1b suggests that a tiny orthogonal history direction can be unusually readable by some frozen downstream geometry. Gate 2 falsifies the first literal implementation of that direction as an explicit cosine retrieval address. Gate 3 still does not achieve absolute retrieval, but on three smoke positions the frozen value receiver maps the real temporal direction into a representation that is markedly more predecessor-specific than matched shuffled directions. None of these results establishes a biological waveform code, consciousness, learned attention, episodic memory, or that a fly connectome is intrinsically suited to chess. Larger held-out position sets are required before treating the Gate 3 percentile pattern as general.