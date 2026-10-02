# Observer Lens Gates Implementation Plan

**Goal:** Extend the existing ChessFlyStatePings experiment sequence with four bounded diagnostics: softmax-gauge-safe listener geometry, full retrieval structure, nonlinear curvature/rectification, and fixed-ping receiver-state crossing, before any recurrent feedback-drift claim.

## Task 1 — Gauge-safe Gate 4
- Center value/policy output signatures row-wise before cosine retrieval.
- Retain uncentered results for comparison.
- Record full similarity matrices, per-query tie-aware ranks, and margins for association/value/policy real listeners.
- Compare centered learned listeners against the same paired column-permuted controls already used by Gate 4.

## Task 2 — Even response / curvature gate
- Add `c(h,r,a) = (A(h+ar)+A(h-ar))/2 - A(h)` in association space.
- Verify a linear decoder yields zero curvature response.
- Verify small-amplitude scaling is quadratic in `a`.
- Where decoder preactivations/weights are exposed, compare the numerical even response to the analytic GELU Hessian-direction prediction.

## Task 3 — Fixed ping × receiver state gate
- Hold one real temporal ping fixed while evaluating it at multiple receiver baselines from the same records.
- Cross state and ping independently: real-state/real-ping, state-swapped real ping, matched control ping, and fixed linear receiver control.
- Measure centered downstream similarity/readout changes and report complete matrices/ranks rather than only means.

## Task 4 — Feedback drift gate
- Only after Tasks 1–3 are inspectable, add an explicitly opt-in recurrent experiment that feeds the even response back into state.
- Compare zero-mean ±ping schedules against a linear/no-feedback control.
- Report accumulated displacement over settling steps and preserve any null result.

## Boundaries
- No retraining.
- Frozen ChessFly checkpoint and graph.
- No claim that effective readout drift is physical drift until feedback is actually enabled.
- No quantum, consciousness, or biological-memory claim.
- Each task gets its own tests and can fail independently.
