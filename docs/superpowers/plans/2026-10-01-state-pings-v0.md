# ChessFly StatePings v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Python research harness that automatically acquires the published ChessFly artifacts from Hugging Face, reproduces the frozen baseline, adds the global fast/slow StatePing recurrence, and evaluates it with paired positions and deterministic headless games.

**Architecture:** Keep upstream artifacts external and cached locally. Separate artifact/provenance handling, graph/weight decoding, baseline recurrence, StatePing recurrence, chess policy/evaluation, arena, and receipt generation so the baseline can be tested independently from the intervention. Use tiny synthetic fixtures for the default test suite; real artifacts are opt-in integration only.

**Tech Stack:** Python 3.11+, PyTorch, safetensors, python-chess, platformdirs, stdlib urllib/json/hashlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-state-pings-v0-design.md`

## Global Constraints

- Do not commit or redistribute `flynet.safetensors`, `connectome.bin.gz`, `neurons.bin.gz`, decompressed graph data, or Hugging Face cache contents.
- Keep graph topology, edge signs/gains, encoder, scale/shift tensors, decoder, policy/value heads, settle depth, and legal masking fixed in v0.
- No retraining in v0.
- Baseline starts each position at `h0=0`; StatePing starts at `h0=s0=0` and does not carry neural state across real chess moves.
- `kappa=0` must be numerically equivalent to baseline within test tolerance.
- NaN/Inf aborts inference and is reported as instability.
- Default StatePing sweep: `rho=[0.25,0.50,0.75,0.90]`, `kappa=[-0.20,-0.10,-0.05,0.0,0.05,0.10,0.20]`.
- Stockfish is optional; raw policy comparison and arena must work without it.
- CLI default device is `auto`: CUDA only when available, else CPU; explicit unavailable device is an error.
- Repository license remains GPL-3.0 and README/attribution must identify ChessFly as upstream work.

## Review Focus

- Interrupted/partial artifact downloads must never be accepted as valid cached files; owning tests live in Task 1.
- Malformed graph headers/indices or unknown group labels must fail before sparse-matrix construction; owning tests live in Task 2.
- An explicit unavailable CUDA device must error instead of silently falling back; owning tests live in Task 3.
- StatePing instability (NaN/Inf) must not reach move selection or produce a legal-looking result; owning tests live in Tasks 3 and 4.
- Arena determinism must survive color pairing, max-ply draws, and policy forfeits; owning tests live in Task 5.

---

### Task 1: Package skeleton, attribution, and verified Hugging Face artifact cache

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `ATTRIBUTION.md`
- Create: `src/chessfly_statepings/__init__.py`
- Create: `src/chessfly_statepings/assets.py`
- Create: `tests/test_assets.py`

**Interfaces:**
- Produces: `ArtifactPaths`, `ArtifactRecord`, `ArtifactManifest`, `default_cache_dir()`, `ensure_artifacts(cache_dir=None, space_revision="main", model_revision="main", force=False) -> ArtifactManifest`.
- Later tasks consume `ArtifactManifest.paths` for graph/weight loading and receipt provenance.

- [ ] **Step 1: Write failing cache/provenance tests**

Tests assert: verified existing files avoid downloader calls; a partial/failed download leaves no destination file; SHA mismatch raises `ArtifactError`; the default cache path is user-local; `.gitignore` excludes artifact names and cache directories.

- [ ] **Step 2: Run `pytest tests/test_assets.py -q`**

Expected: FAIL because package/artifact API does not exist.

- [ ] **Step 3: Implement package metadata, attribution, ignore rules, and atomic acquisition**

Use `urllib.request` with temporary files + `os.replace`; read Space `data/meta.json`; record requested revisions, source URLs, sizes, compressed hashes, decoded hashes when published, and a UTC timestamp. Do not depend on `huggingface_hub` for v0.

- [ ] **Step 4: Run `pytest tests/test_assets.py -q && pytest -q`**

Expected: PASS.

- [ ] **Step 5: Commit**

`git commit -m "feat: add verified ChessFly artifact acquisition"`

### Task 2: Chess encoding, packed graph parser, and weight validation

**Files:**
- Create: `src/chessfly_statepings/encoding.py`
- Create: `src/chessfly_statepings/graph.py`
- Create: `src/chessfly_statepings/weights.py`
- Create: `tests/test_encoding.py`
- Create: `tests/test_graph.py`
- Create: `tests/test_weights.py`

**Interfaces:**
- Consumes: artifact paths from Task 1.
- Produces: `ACTION_SPACE`, `ACTION_INDEX`, `canonical_fen()`, `mirror_uci()`, `encode_fen()`, `ChessFlyGraph.from_files()`, `ChessFlyGraph.sparse_matrix()`, `ChessFlyWeights.from_file()`.

- [ ] **Step 1: Write failing encoding tests**

Assert deterministic 1,968-action space, canonical black-to-move mirroring, promotion preservation, 780-feature layout, and castling/en-passant flags.

- [ ] **Step 2: Write failing graph tests**

Using tiny binary fixtures, assert source-CSR to target-CSR transpose, sign preservation, input/readout group extraction, bounds rejection, malformed size rejection, and unknown group rejection.

- [ ] **Step 3: Write failing weight tests**

Using tiny in-memory/safetensors fixtures, assert required tensor names, compatible shapes, checkpoint metadata parsing (`steps`, `alpha`, `hidden`, optional `step`), and clear errors for mismatches.

- [ ] **Step 4: Run `pytest tests/test_encoding.py tests/test_graph.py tests/test_weights.py -q`**

Expected: FAIL because modules are missing.

- [ ] **Step 5: Implement encoding, graph, and weight modules**

Follow the public ChessFly contract exactly; no search or StatePing logic belongs here.

- [ ] **Step 6: Run targeted tests and full `pytest -q`**

Expected: PASS.

- [ ] **Step 7: Commit**

`git commit -m "feat: decode ChessFly graph weights and chess inputs"`

### Task 3: Exact baseline recurrence, trajectory probe, and StatePing model

**Files:**
- Create: `src/chessfly_statepings/model.py`
- Create: `src/chessfly_statepings/state_ping.py`
- Create: `tests/test_model.py`
- Create: `tests/test_state_ping.py`

**Interfaces:**
- Consumes: `ChessFlyGraph`, `ChessFlyWeights`, encoded 780-vectors.
- Produces: `ForwardResult(policy_logits, value_logits, activity, instability)`, `ChessFlyBaseline.forward(features, include_activity=False)`, `StatePingModel.forward(features, rho, kappa, include_activity=False)`, `trajectory_summary(activity, rho, readout_indices)`.

- [ ] **Step 1: Write failing baseline tests**

On a tiny known graph/tensor fixture, analytically check each settle step, final readout, policy/value tensors, batching, device resolution, and explicit unavailable CUDA error.

- [ ] **Step 2: Write failing residue/StatePing tests**

Assert analytical slow/residue sequences for fixed `rho`; `kappa=0` equals baseline; positive and negative `kappa` alter recurrence while graph arrays/signs remain unchanged; full and compact trajectory summaries agree; NaN/Inf returns instability/raises the declared inference exception before policy output is used.

- [ ] **Step 3: Run `pytest tests/test_model.py tests/test_state_ping.py -q`**

Expected: FAIL because runtime does not exist.

- [ ] **Step 4: Implement baseline and StatePing recurrence**

Baseline recurrence is isolated. StatePing computes `r_t` from the previous/current settle trajectory and uses `W h + kappa W r`; all other checkpoint transforms remain identical.

- [ ] **Step 5: Run targeted tests and full `pytest -q`**

Expected: PASS.

- [ ] **Step 6: Commit**

`git commit -m "feat: add baseline and StatePing recurrent dynamics"`

### Task 4: Legal policy, paired-position comparison, optional Stockfish, and receipts

**Files:**
- Create: `src/chessfly_statepings/policy.py`
- Create: `src/chessfly_statepings/compare.py`
- Create: `src/chessfly_statepings/receipts.py`
- Create: `data/smoke_fens.txt`
- Create: `tests/test_policy.py`
- Create: `tests/test_compare.py`
- Create: `tests/test_receipts.py`

**Interfaces:**
- Consumes: baseline/StatePing forward results and Task 1 artifact manifest.
- Produces: `PolicyDecision`, `ChessFlyPolicy.select(board)`, `compare_positions(...) -> CompareResult`, optional `StockfishScorer`, `write_receipt(path, payload)`.

- [ ] **Step 1: Write failing policy tests**

Assert legal masking outside the graph, deterministic greedy argmax, correct black mirroring, top-5 probabilities, value expectation, and instability refusal.

- [ ] **Step 2: Write failing paired-comparison tests**

Assert identical FEN inputs feed both models, move-change flag, Jensen-Shannon divergence, value delta, compact activity statistics, and optional Stockfish absence is non-fatal.

- [ ] **Step 3: Write failing receipt tests**

Assert timestamp, command/config, package versions, device, artifact hashes/revisions, model metadata, input identifiers, aggregate metrics, and instability counters are serialized as small JSON.

- [ ] **Step 4: Run targeted tests**

Expected: FAIL because APIs are missing.

- [ ] **Step 5: Implement policy, comparison, optional engine wrapper, receipts, and smoke FENs**

- [ ] **Step 6: Run targeted tests and full `pytest -q`**

Expected: PASS.

- [ ] **Step 7: Commit**

`git commit -m "feat: add paired ChessFly evaluation and receipts"`

### Task 5: Deterministic headless arena, CLI, sweep command, and opt-in real-artifact integration

**Files:**
- Create: `src/chessfly_statepings/arena.py`
- Create: `src/chessfly_statepings/cli.py`
- Create: `src/chessfly_statepings/__main__.py`
- Create: `tests/test_arena.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_integration_real.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all prior public APIs.
- Produces CLI commands: `assets`, `probe`, `compare`, `sweep`, `arena`; `play_paired_arena(...) -> ArenaResult`.

- [ ] **Step 1: Write failing arena tests**

With toy deterministic policies, assert paired colors, seeded openings, max-ply draw handling, W/D/L and score aggregation, mean game length, move-disagreement tracking, and instability/exception forfeits.

- [ ] **Step 2: Write failing CLI tests**

Assert every command parses; `probe/compare/sweep/arena` automatically call artifact acquisition through an injectable loader; explicit cache/device/rho/kappa/seed/output options propagate; `sweep` retains every declared setting.

- [ ] **Step 3: Write opt-in integration test marker**

When `CHESSFLY_RUN_INTEGRATION=1`, download/verify real artifacts, load the real graph/checkpoint, assert dimensions agree, and assert two baseline forwards for one pinned FEN produce identical logits. Default suite skips it.

- [ ] **Step 4: Run targeted tests**

Expected: FAIL because arena/CLI do not exist.

- [ ] **Step 5: Implement arena, CLI, sweep orchestration, and README usage**

Default arena is raw one-forward greedy policy; no chess search is added in v0. README documents first-run downloads, cache location, provenance, scientific boundaries, example commands, and how to opt into Stockfish/integration.

- [ ] **Step 6: Run `pytest -q`**

Expected: all default tests PASS; real integration SKIPPED unless explicitly enabled.

- [ ] **Step 7: Run local smoke commands without real artifacts**

`python -m chessfly_statepings --help` and CLI parser/unit-fixture smoke must succeed.

- [ ] **Step 8: Commit**

`git commit -m "feat: add headless arena and StatePing CLI"`

### Task 6: Final verification and branch readiness

**Files:**
- Modify only if verification exposes defects.

**Interfaces:**
- Consumes entire project.
- Produces a branch that is internally verified and ready for user-run real-artifact experiments.

- [ ] **Step 1: Run `pytest -q` and record exact result**

Expected: PASS with only the opt-in integration test skipped.

- [ ] **Step 2: Run packaging/import checks**

`python -m pip install -e . --no-deps`, `python -m chessfly_statepings --help`, and import all public modules.

- [ ] **Step 3: Inspect git diff/status for accidentally tracked large/binary artifacts**

Expected: no model/connectome/cache files; only source, docs, smoke text, tests, and plan/spec.

- [ ] **Step 4: Whole-branch review against spec and Review Focus**

Critical/Important findings receive one RED→GREEN fix pass plus a green full suite; Minor findings are documented.

- [ ] **Step 5: Commit any verification fixes**

`git commit -m "fix: harden StatePings v0 verification"` only if needed.
