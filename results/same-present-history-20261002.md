# Same present, retained history, fixed delayed ping

**Result: the receiver responses differ with retained history, but correctly targeted continuation fails the declared gate.** The primary policy assay gets 13/24 pairings correct (54.2%), a 68.2% matched-control margin percentile and negative average alignment with its intended continuation.

The question was: **With the same present input, do different retained histories make the same delayed ping produce different, correctly targeted continuations?** This experiment separates “different” from “correctly targeted.”

## What was measured

The frozen ChessFly checkpoint and graph run 12 declared cue pairs across four opening positions. For each case, the two receivers see A,B and B,A, then the same common-present input from step 3 onward. Their cue bags are identical; only order differs. These are controlled sensory snapshots inside a five-step settle, not legal move sequences or state carried between games.

A present-only third trajectory constructs one common orthogonal fast-minus-slow ping from steps 1–3 at rho=0.75. It is delivered at step 4, after two identical-present inputs. Both histories receive exactly the same ping, without individual re-orthogonalization. The target is each history's own untouched step-4 to step-5 transition, which is withheld from ping construction.

For the frozen decoder D, the response is

\[
g_i(a)=\frac{D(h_i+a p)-D(h_i-a p)}{2a},\qquad
\delta_i=D(h_{i,\mathrm{next}})-D(h_i).
\]

Each response is compared with **both** target directions, using complete 2x2 cosine matrices. Policy logits are restricted to the common present's legal moves and centered. Value logits are also centered. Association and value geometry are secondary. No query direction, sign, magnitude or target is fitted to the result.

Controls erase the cue prefix, swap response/target pairings, use the state-blind affine decoder Wp, and replace the ping with 32 shuffled, re-orthogonalized, norm-matched directions. Each control ping is shared across both histories and magnitudes. The default seed is 0. Ties within 1e-6 get half credit. Unidentifiable response or target directions yield explicit inconclusive results.

## Primary result: delay 2, magnitude 1

| Measure | Observed | Required |
| --- | ---: | ---: |
| Identifiable case pairs | 12 | at least 12 |
| Native pairing accuracy | 54.17% (13/24) | at least 75% |
| Mean native minus crossed cosine | +0.00009613 | greater than 0.000001 |
| Mean native continuation cosine | -0.010663 | greater than 0 |
| Margin improvement over erased history | +0.00009613 | greater than 0.000001 |
| Matched-control margin percentile | 68.18% | at least 95% |
| Gate | **failed** | all criteria must pass |

Erased-history and linear-reader margins are effectively zero. Swapping the retained responses reverses the real margin to -0.00009613. Those controls behave as intended; the small positive real pairing margin is insufficient for success. A response can be slightly closer to its native target than to the crossed target while still pointing away from both.

All 12 readout states remain distinct after the shared input, with pair separations ranging from 4,241 to 8,305 in raw model units. The fixed ping norms are 0.215–0.234. Mean relative policy-response separation is 0.233: history-dependent response differences are measurable. These raw scale and separation measurements are not correctness scores.

| Magnitude | Policy accuracy | Native margin | Native cosine | Control percentile |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | 54.17% | +0.00000344 | -0.010704 | 65.15% |
| 1 (primary) | 54.17% | +0.00009613 | -0.010663 | 68.18% |
| 2 | 54.17% | +0.00023502 | -0.010462 | 71.21% |

The secondary primary-magnitude association accuracy is 41.67%, with a negative native pairing margin (-0.001062). Value accuracy is 50%, with negative pairing margin (-0.001044) and strongly negative native alignment (-0.7882). The sensitivity settings do not establish the missing claim.

## Preserved first attempt and timing correction

Version 1 constructed the ping from only the first two present-only steps and tested delays 1 and 2. Its first receipt is **inconclusive**, with zero identifiable cases in every geometry and setting. Ping norms were 0 to 1.33e-7.

A source-only diagnostic explains this limitation: for all four common inputs, the step-1 readout is exactly zero. The two-step residue is therefore rho times the step-2 readout in real arithmetic. Removing the current-state direction must remove the entire residue. The tiny surviving values are floating-point rounding residue.

The version 2 [amendment](../docs/superpowers/specs/2026-10-02-same-present-history-design.md#version-2-amendment-source-timing-before-its-measurement) was committed before its run. It waits for the earliest noncollinear reference prefix, step 3, delivers at step 4, and retains step 5 as a held-out target. That leaves only one causal delay in the original five-step model. Cases, magnitudes, rho, seed, controls and thresholds stayed fixed. Both first receipts are preserved; version 2 is a follow-up after observing a source limitation, not an untouched first protocol.

## Reproduction and provenance

```bash
chessfly-statepings history-lens --cases data/history_lens_cases.json --device cpu --rho 0.75 --seed 0 --controls 32 --ping-steps 3 --delays 2 --magnitudes 0.5 1 2 --output results/history-lens-v2-rerun.json
```

To reproduce version 1, use `--ping-steps 2 --delays 1 2`. Both protocols keep the checkpoint's original five steps. Querying reads the receiver at the delivery time; it does not feed the probe response back into recurrence.

- [Version 2 full receipt](receipts/history-lens-v2-20261002.json), code `e269343c14b0572ecfb868335ca23e6727d598a3`.
- [Compact summary](receipts/history-lens-v2-20261002-summary.json), including the full receipt SHA-256.
- [Version 1 first receipt](receipts/history-lens-20261002.json), code `555a6211eb82e9a310b437974fa34dc9ba124949`.
- [Source-only diagnostic](receipts/history-lens-source-diagnostic-20261002.json).

The [code-history bundle](provenance/README.md) preserves the exact local commits named in these receipts; the GitHub publication commit has different metadata. The reviewed code and raw receipt bytes are unchanged.

Both real runs used CPU, PyTorch 2.7.1+cpu and four threads, with zero instabilities. The weights SHA-256 is `e05a675c35726b6784c563f6c29ab3fe6afac249e0b143560286a81eab9380d5`; it matches the earlier supplied CUDA receipts. Artifact revisions were requested as `main`; resolved revision headers were unavailable, so recorded hashes are the precise artifact identifiers. Reproducibility across devices is subject to floating-point differences, particularly for small margins.

The complete suite, including the real-artifact deterministic baseline integration test, passed: 116 tests. Source-timing regressions include the zero-first-readout case, a delayed-delivery check and tests that future inputs cannot alter the source prefix or earlier receiver states.

Independent review reproduced every version 2 result exactly and reproduced the version 1 runs and primary result. A decoder-only float64 diagnostic, keeping the float32 recurrence and ping fixed, changed the small primary native margin from about 0.0000961 to 0.0002264; mean relative policy-response error was 0.73%. Accuracy stayed 13/24, native alignment stayed negative and substantial response separation remained. Small margins are precision-sensitive; this diagnostic does not overturn the failed gate, and the original first receipts are unchanged.

## Claim boundary and the motivating paper

“Correct” means this model's next unmodified internal transition. It does not mean the best chess move or a beneficial intervention. The receiver's current hidden state carries retained history; this assay does not establish memory beyond that complete state. The 12 pairs share four common positions, and their 24 queries are correlated; the reported percentiles and accuracies are descriptive, not significance tests or evidence of generality.

Martin-Burgos et al., [*Action potential waveforms are state-dependent*](https://doi.org/10.64898/2026.09.15.751814), bioRxiv, supplied version posted 21 September 2026, motivates the distinction between a state-dependent outgoing signal and its meaning for a receiver. The preprint reports waveform relationships with drive and heterogeneous relationships with LFP state; it does not demonstrate the continuation-specific receiver decoding tested here. ChessFly has artificial recurrent units, not biophysical action potentials, cortical tufts or a Martinotti circuit.

This assay establishes measurable history-dependent responses in the tested frozen model. It does not demonstrate that the common ping usefully selects the continuation native to each history.
