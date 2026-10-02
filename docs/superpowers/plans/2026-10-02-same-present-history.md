# Same-Present History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a reproducible fixed-ping assay of history-dependent, target-specific internal continuation under identical present input.

**Architecture:** An experimental input-schedule runner reuses the frozen recurrence without changing baseline APIs. A focused evaluator generates fixed causal pings, compares two retained states with held-out future transitions and controls, and exports full matrices through the existing CLI/receipt system.

**Tech Stack:** Existing Python >=3.11, PyTorch >=2.0, NumPy >=1.26, python-chess >=1.999 and pytest >=8; no new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-10-02-same-present-history-design.md`

## Global Constraints

- Frozen graph/checkpoint; no training or state carry between actual chess moves.
- Prefix A,B versus B,A; common present begins at step 3.
- Fixed ping uses only present-only steps 1–2; default rho=0.75, seed=0, controls=32.
- Default delays 1,2; magnitudes 0.5,1,2; primary delay=1/magnitude=1.
- Primary geometry is centered legal-policy logits; at least 12 identifiable cases, accuracy >=0.75, margin >1e-6, positive native alignment, improvement over erased history >1e-6, control percentile >=0.95.
- Preserve negative/inconclusive outcomes and all matrices. No paper PDF or upstream weights in git.

**Version 2 amendment:** The first real run exposed a structurally zero two-step readout ping; its receipt and source-only diagnostic are preserved. Before the second run, the spec adds a three-step reference prefix, delivery at step 4 and target at step 5. Version 2 defaults to delay 2 with primary magnitude 1; all other controls and criteria remain fixed. Task 3 includes the causal timing correction, its RED/GREEN regression and a separately versioned first receipt. The original protocol remains reproducible with `--ping-steps 2 --delays 1 2`.

## Review Focus

- A changing final schedule step must change targets but leave earlier states and the fixed ping unchanged.
- Identical-present/erased histories and linear readers must not receive artificial native-pair credit.
- Zero pings/targets and indistinguishable target directions must become inconclusive, with finite JSON.
- The same fixed/control ping must be shared across both histories, including after delays.
- Invalid cases/delays must fail before emitting a scientifically misleading success receipt.

### Task 1: Causal schedules and behavioral falsifiers

**Files:** Create `src/chessfly_statepings/history_lens.py`; test `tests/test_history_lens.py`.

**Interfaces:** `scheduled_activity(model, features: Tensor) -> tuple[Tensor,...]` consumes [steps,batch,780] inputs. `paired_continuation_scores(responses, targets)` returns complete two-candidate scores and tie-aware metrics.

- [x] Write tests with hand-derived recurrence states for ordered versus reversed cues and a shared suffix, an identical-input baseline equivalence check, future-only changes, nonlinear/state-blind reader comparisons, ties and zero targets.
- [x] Run the tests and confirm missing feature failures.
- [x] Implement the schedule runner and pair scoring without mutating model tensors or baseline methods.
- [x] Run the focused tests; commit the independently usable schedule/scoring unit.

### Task 2: Case evaluation and controls

**Files:** Extend `history_lens.py`; create `data/history_lens_cases.json`; extend `tests/test_history_lens.py`.

**Interfaces:** `load_history_cases(path)` supplies validated case records; `evaluate_history_lens(cases, model, *, rho=.75, delays=(1,2), magnitudes=(.5,1,2), controls=32, seed=0)` supplies serializable runs, per-case metrics and primary decision.

- [x] Write failing tests for the fixed-ping causal boundary, ordered cue multiset, erase/swap/state-blind controls, nonfinite inputs and unidentifiable targets.
- [x] Implement case validation, predeclared data, batched decoder probes and controls, aggregate criteria and explicit inconclusive status.
- [x] Run focused tests and verify the positive synthetic falsifier and negative linear/reset cases; commit.

### Task 3: CLI, real-run receipts and documentation

**Files:** Modify `src/chessfly_statepings/cli.py`, `README.md`, `results/README.md`, `site/index.html`; create `tests/test_history_lens_cli.py` and real receipts if upstream artifacts are reachable.

- [x] Write a CLI behavioral test using the real tiny model and a local manifest; confirm the missing command fails.
- [x] Register `history-lens --cases ... --delays ... --magnitudes ... --controls ... --seed ... --rho ...` and write the existing receipt schema.
- [x] Run the full repository suite, compile checks and diff checks.
- [x] Run the real-artifact assay if available; preserve the first receipt and document the result without tuning.
- [x] Request an independent whole-change review, resolve findings, push the branch and open a stacked PR against `feature/observer-lens-gates`.

Published as https://github.com/anttiluode/ChessFlyStatePings/pull/9, stacked on PR 8. The targeting gate failed; all 116 tests passed, including real-artifact integration. Independent review reproduced both runs without blocking findings. Exact measured commits are preserved in the verified provenance bundle.
