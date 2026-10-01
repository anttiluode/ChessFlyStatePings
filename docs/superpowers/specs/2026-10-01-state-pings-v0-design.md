# ChessFly StatePings v0 Design

## Purpose

`ChessFlyStatePings` is an experimental derivative and research harness around the published ChessFly model. Its goal is not to reproduce or redistribute the original project as if it were ours, nor to claim biological fidelity. The goal is to ask a narrower computational question inspired by `BrainAsInverseModelerV3`:

> If the same fixed connectome and trained ChessFly checkpoint are kept intact, does exposing a small history-dependent component of each unit's recent trajectory change useful computation through the graph?

The experiment must make the original ChessFly computation the zero-change control, keep large upstream artifacts outside GitHub, and produce machine-readable receipts that distinguish observation from intervention.

## Provenance and licensing boundary

This repository is GPL-3.0 and will clearly attribute Maxime Labonne's ChessFly, the FlyWire-derived connectome sources cited by ChessFly, and the BrainAsInverseModelerV3 motivation.

The repository will **not** commit or redistribute:

- `flynet.safetensors`;
- `connectome.bin.gz`;
- `neurons.bin.gz`;
- decompressed connectome artifacts;
- any Hugging Face cache containing those files.

Instead, the runtime downloads the required artifacts directly from the original Hugging Face locations on first use and caches them locally. The acquisition layer records source repository, requested revision, resolved revision when available, byte size, and SHA-256. The graph files are additionally checked against the decoded SHA-256 values published by ChessFly metadata. The trained model is treated as an external artifact whose upstream terms remain authoritative; this repository contains only code, attribution, small test fixtures, and experiment receipts.

Default upstream sources:

- model repo: `mlabonne/chessfly`;
- Space repo: `mlabonne/chessfly`, `repo_type="space"`;
- checkpoint: `flynet.safetensors`;
- graph: `data/connectome.bin.gz`;
- neuron metadata: `data/neurons.bin.gz`.

The current public model checkpoint corresponds to the published step-12,000 FlyNet release. Release-quality receipts must pin immutable Hugging Face revisions; development commands may resolve `main` and record the resolved commit.

## Scientific boundaries

The ChessFly units are artificial recurrent units wired by a published fly-derived graph. They are not biophysical neuron simulations. Terms such as "state-bearing ping" describe a computational intervention, not evidence that Drosophila neurons transmit the same code.

The v0 experiment keeps all of the following fixed unless a named experiment explicitly says otherwise:

- graph topology;
- excitatory/inhibitory edge signs;
- learned per-edge gains;
- board encoder;
- original per-step scale/shift tensors;
- decoder, policy head, and value head;
- five-step settling depth;
- legal-move masking.

No retraining is part of v0. The first intervention is deliberately parameter-free apart from two scalar hyperparameters (`rho`, `kappa`).

## Baseline compatibility target

The baseline Python runtime must reproduce the public ChessFly forward contract:

1. encode a canonical white-perspective board into 780 features;
2. inject the learned encoder output at input neurons on every recurrent step;
3. initialize hidden activity to zero for every board position;
4. run exactly the checkpoint-declared number of recurrent steps (currently 5);
5. apply the original sparse recurrent graph, per-step scale/shift, leak update, and ReLU;
6. read the declared central/descending readout neurons;
7. emit 1,968 policy logits and 64 value logits;
8. apply legal-move masking outside the neural graph.

The baseline path must be isolated from the StatePing path so a regression test can establish that `kappa=0` is numerically equivalent to the baseline within a declared tolerance.

## Local artifact cache

The default cache is platform-neutral and user-local, resolved with `platformdirs`, e.g.:

```text
<user-cache>/chessfly-statepings/
    artifacts/
        flynet.safetensors
        connectome.bin.gz
        neurons.bin.gz
        manifest.json
```

`ensure_artifacts()` is idempotent. If verified artifacts already exist, no network access is required. Partial downloads must use temporary files and atomic rename. A failed hash check is fatal and must never silently reuse the file.

The Git repository ignores all artifact and cache paths.

## Runtime architecture

The package is `chessfly_statepings` and is split by responsibility:

- `assets.py` — Hugging Face acquisition, cache paths, revision/hash manifest, verification;
- `graph.py` — packed graph/neuron file parsing and sparse PyTorch matrix construction;
- `encoding.py` — ChessFly board encoding, canonical mirroring, and 1,968-action UCI mapping;
- `weights.py` — safetensors loading and checkpoint metadata validation;
- `model.py` — exact baseline forward pass and shared output dataclasses;
- `state_ping.py` — trajectory extraction and modified StatePing dynamics;
- `policy.py` — legal move selection and value conversion;
- `arena.py` — deterministic headless model-vs-model games and PGN/result summaries;
- `receipts.py` — JSON provenance/config/result receipts;
- `cli.py` — user-facing entry points.

Tests use small synthetic graphs and tensor fixtures. Full-artifact integration tests are opt-in because the real upstream assets are large.

## Observation gate: trajectory probe

Before changing ChessFly dynamics, v0 must be able to observe the five original hidden states

\[
h_1, h_2, \ldots, h_K.
\]

For a scalar `rho` with `0 <= rho < 1`, define a local slow state

\[
s_0 = 0,
\]

\[
s_{t+1}=\rho s_t+(1-\rho)h_{t+1},
\]

and residue

\[
r_{t+1}=h_{t+1}-s_{t+1}.
\]

The trajectory probe is observational only. It must return compact summary statistics by default (mean absolute residue, RMS residue, active fraction, cosine between `h` and `r`, readout-region equivalents) and optionally expose full tensors to Python callers. CLI receipts must not dump full 138k-neuron states.

The first probe asks whether nontrivial fast/slow trajectory structure exists under the frozen network. It makes no performance claim by itself.

## StatePing intervention

The v0 StatePing model augments the recurrent message using the same sparse graph:

\[
q_t = W h_t,
\]

\[
q^{(r)}_t = W r_t,
\]

\[
q'_t = q_t + \kappa q^{(r)}_t.
\]

The original input drive, scale/shift, ReLU, and leak update then operate on `q'_t` exactly as in the baseline.

Properties required by design:

- `kappa = 0` is the exact baseline control;
- no edge is added, removed, or sign-flipped;
- `rho` and `kappa` are global scalars in v0, not per-edge learned parameters;
- the slow state exists only within the K-step settle for one board position in v0;
- every new board position still starts from `h_0 = 0`, `s_0 = 0` so ordinary chess history is not yet introduced;
- positive and negative `kappa` are both supported;
- NaN/Inf activity aborts the forward pass and is recorded as instability rather than coerced into a chess move.

The initial declared sweep is:

```text
rho   = [0.25, 0.50, 0.75, 0.90]
kappa = [-0.20, -0.10, -0.05, 0.0, +0.05, +0.10, +0.20]
```

This sweep is exploratory. No claim is based only on picking the best value after seeing results; receipts retain every tested setting.

## Paired-position evaluation

Game outcomes are too noisy to be the first metric. The primary v0 comparison is paired inference over the same FEN positions.

For each FEN, baseline and StatePing receive identical board input and produce:

- selected legal move;
- top-5 legal move probabilities;
- value expectation;
- policy KL/Jensen-Shannon divergence;
- whether the selected move changed;
- activity/residue stability summaries.

If a local Stockfish executable is available, the evaluator may additionally record:

- engine evaluation of baseline selected move;
- engine evaluation of StatePing selected move;
- centipawn-loss difference;
- mate-status changes.

Stockfish is optional. The project must remain runnable without it.

The built-in smoke dataset is a small, hand-auditable set of FENs committed as text. Larger benchmark datasets are user-supplied or downloaded by a separate optional command with their own provenance.

## Headless arena

The arena plays deterministic games between two policy configurations using `python-chess`.

For each opening seed, colors are paired:

1. baseline as White vs StatePing as Black;
2. StatePing as White vs baseline as Black.

Default move selection is greedy legal argmax from one network forward pass. This is the cleanest neural comparison. A later optional `--search-depth` mode may implement the same shallow search wrapper for both players, but raw policy play is the v0 default.

Arena safeguards:

- deterministic opening list and RNG seed;
- maximum ply limit;
- normal python-chess terminal/draw rules;
- invalid or unstable model output forfeits the game and is recorded explicitly;
- no hidden state persists between real chess moves in v0;
- PGN may be written locally, while the compact JSON receipt is the canonical result artifact.

Reported arena metrics:

- W/D/L by side and aggregate;
- score (`win=1`, `draw=0.5`, `loss=0`);
- move disagreement rate between baseline and StatePing on positions encountered;
- mean game length;
- instability/forfeit count.

Game results are descriptive, not an Elo estimate unless a later experiment is explicitly designed for rating inference.

## CLI contract

The intended first-run experience is:

```bash
python -m pip install -e .
chessfly-statepings assets
chessfly-statepings probe --fen "<FEN>"
chessfly-statepings compare --positions data/smoke_fens.txt --rho 0.75 --kappa 0.10
chessfly-statepings arena --games 20 --rho 0.75 --kappa 0.10
```

`probe`, `compare`, and `arena` call `ensure_artifacts()` automatically, so the explicit `assets` command is optional. `--device auto` selects CUDA only when PyTorch reports it available; otherwise CPU. An explicit unavailable device is an error rather than silently changing the requested device.

## Receipts and reproducibility

Every nontrivial run writes a JSON receipt containing at least:

- repository version / Git commit when discoverable;
- UTC timestamp;
- command and arguments;
- Python, PyTorch, and python-chess versions;
- device and thread configuration;
- artifact source repos, requested/resolved revisions, byte sizes, SHA-256 hashes;
- model checkpoint metadata (`step`, `steps`, `alpha`, dimensions when present);
- `rho`, `kappa`, and search settings;
- input FEN/opening identifiers and seed;
- aggregate results;
- instability counters.

Receipts are small and may be committed. Large PGNs, caches, tensors, and upstream artifacts are ignored by Git.

## Testing strategy

Unit tests must cover:

1. FEN canonicalization and action-space determinism;
2. graph parser bounds, source-to-target transpose, signs, and group extraction;
3. safetensors metadata/shape validation using tiny fixtures;
4. atomic artifact cache behavior and hash mismatch rejection using mocked/local downloads;
5. baseline recurrent update on a tiny known graph;
6. exact `kappa=0` equivalence between baseline and StatePing paths;
7. analytically checkable residue dynamics for chosen `rho` values;
8. positive/negative `kappa` altering recurrence without changing graph topology/sign data;
9. NaN/Inf instability detection;
10. deterministic move selection and mirrored black-to-move handling;
11. deterministic paired-color arena behavior on toy policies;
12. JSON receipt schema essentials.

An opt-in integration test downloads the real artifacts and verifies graph/checkpoint compatibility plus baseline deterministic logits for a pinned FEN. It is not part of the fast default test suite.

## v0 success and falsification

The engineering success criterion is straightforward: a fresh clone can install, acquire/verify the external artifacts automatically, reproduce the baseline path, run StatePing variants, evaluate paired positions, and play headless paired games while generating reproducible receipts.

The scientific outcome is deliberately open:

- If every nonzero `kappa` degrades or destabilizes the model, the intervention failed under the frozen checkpoint.
- If small `kappa` changes decisions but not quality under engine-labelled comparisons, StatePing exposes sensitivity but no demonstrated advantage.
- If a predeclared setting or a later independently confirmed setting improves held-out engine-labelled move quality/value prediction without instability, that motivates a stricter follow-up.
- Headless match wins alone are insufficient to claim improvement.

No v0 result establishes a biological waveform code, consciousness, or that the fly connectome is intrinsically suited to chess.

## Deferred work

Explicitly out of scope for v0:

- retraining the 15M learned edge gains;
- per-neuron or per-synapse learned StatePing readout coefficients;
- preserving neural state across real chess moves;
- repetition/history-sensitive chess tasks;
- WebGPU/browser UI modifications;
- Elo estimation;
- claiming equivalence between artificial activation residues and action-potential waveform coding.

Those become separate gates only if v0 produces a reproducible reason to continue.
