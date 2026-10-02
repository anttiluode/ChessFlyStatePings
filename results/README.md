# ChessFlyStatePings results

This directory keeps small, inspectable receipts from real-artifact runs. The model checkpoint and fly-derived graph stay frozen; receipts record the tested intervention, exact inputs, versions, artifact hashes, and measured outcomes.

## 2026-10-02 — Same present, retained history, fixed delayed ping

The [full analysis](same-present-history-20261002.md) tests 12 cue pairs under four shared present positions. A,B and B,A prefixes lead into identical present inputs; the same causal ping is delivered at step 4 and scored against untouched step 5.

**The targeting gate failed.** Histories and responses remain distinct, but primary policy pairing is 13/24 (54.2%), the matched-control margin percentile is 68.2%, and mean native alignment is negative (-0.01066). Association and value results do not establish targeting either. Correctness is internal next-step alignment, not better chess play.

The original two-step source produced an effectively zero orthogonal ping and an inconclusive first run. A source-only diagnostic found a zero first-step readout, making that residue mathematically parallel to current activity. A separately versioned timing correction was committed before its run; both first receipts are retained.

Full receipt: [`receipts/history-lens-v2-20261002.json`](receipts/history-lens-v2-20261002.json). Compact summary: [`receipts/history-lens-v2-20261002-summary.json`](receipts/history-lens-v2-20261002-summary.json).

## 2026-10-01 — Gate 1b directional specificity smoke

The three-position CUDA specificity rerun used 32 shuffled, re-orthogonalized, norm-matched controls per position at magnitudes 1, 2, and 4. Only about 3.03% of the residue norm survived orthogonalization against the present state (about 0.0929% squared energy), yet the real temporal direction ranked unusually high for some frozen readers: mean value-logit sensitivity percentile was about 94.4% at magnitudes 1 and 2, association-layer sensitivity about 86.4%, and policy-logit sensitivity about 68.2%.

This is evidence that the tested temporal direction is not generic to every downstream reader. It is preliminary (n=3) and is not a chess-strength result or a biological spike-code claim.

Receipt: `receipts/specificity-smoke-20261001-summary.json`

## 2026-10-01 — Gate 2 query-memory smoke

Gate 2 made the query metaphor literal. For each of the same three positions, the memory bank stored the penultimate settling state and its orthogonal history coordinate. The final state-bearing ping then tried to retrieve its own predecessor using fixed cosine similarity.

Three arms were compared:

- `present`: current-state coordinate only;
- `history`: equal-weight average of current-state cosine and real orthogonal-history cosine;
- `shuffled_history`: the same present state with another position's history coordinate attached.

### Result

| arm | top-1 accuracy | MRR | mean correct-vs-best-distractor margin |
| --- | ---: | ---: | ---: |
| present | 1.000 | 1.000 | +0.006764 |
| history | 0.333 | 0.611 | -0.006120 |
| shuffled history | 1.000 | 1.000 | +0.005393 |

The literal Gate 2 construction **failed**. Adding the real history coordinate made retrieval substantially worse than present state alone and worse than the shuffled-history control. The mean correct margin changed sign: with real history, the intended memory was on average below the strongest distractor.

That is useful narrowing. Gate 1b showed that a tiny orthogonal history direction can be unusually readable by the frozen decoder/value geometry, but Gate 2 shows that this does not automatically make that coordinate a useful cosine address back to its own recorded settling trajectory. "History is readable" and "history is a retrieval key" are different claims.

Raw receipt: `receipts/query-memory-smoke-20261001.json`

Summary: `receipts/query-memory-smoke-20261001-summary.json`

## 2026-10-02 — Gate 3 receiver-query geometry smoke

Gate 3 tests the narrower hypothesis suggested by Gate 2's failure: perhaps the raw history coordinate is not itself an address, but the **effect it produces inside an already-trained receiver** is more trajectory-specific.

For a readout state `h`, orthogonal history direction `r_perp`, and magnitude `a`, the receiver signature is the sign-symmetric central difference

```text
g(a) = (A(h + a r_perp) - A(h - a r_perp)) / (2a)
```

where `A` is ChessFly's frozen learned decoder association map. The same perturbation is also propagated into the frozen value head. Step 4 supplies the memory signature and step 5 the query signature. Thirty-two controls use the same random permutation at both steps before re-orthogonalization and norm matching.

### Absolute retrieval remained negative

| representation | top-1 accuracy | MRR | mean margin |
| --- | ---: | ---: | ---: |
| raw history | 0.000 | 0.389 | -0.02204 |
| present receiver | 0.333 | 0.611 | -0.07089 |

The real receiver signatures also stayed at 0.333 top-1 accuracy for every tested magnitude. Therefore Gate 3 did **not** establish successful history retrieval.

### But the value receiver selectively reshaped the real history direction

| magnitude | value real margin | matched-control mean margin | margin percentile | MRR percentile |
| ---: | ---: | ---: | ---: | ---: |
| 1 | -0.00589 | -0.19810 | 86.4% | 90.9% |
| 2 | -0.00408 | -0.20093 | 86.4% | 87.9% |
| 4 | -0.00163 | -0.20477 | 86.4% | 89.4% |

This is the main Gate 3 signal. The real value-space history signature remains on the wrong side of zero, but it is far closer to retrieving the correct predecessor than the matched directions are. The effect is stable across all three predeclared magnitudes. Association-space enrichment is weaker: its correct-margin percentile falls from 83.3% at magnitude 1 to 71.2% at 2 and 62.1% at 4.

The current interpretation is therefore deliberately two-part:

1. **No absolute retrieval claim.** Top-1 remains 1/3 and every real margin is negative.
2. **Suggestive receiver-geometry claim.** The already-trained value receiver maps the real temporal direction into a representation that preserves predecessor identity substantially better than matched shuffled directions on this n=3 smoke test.

That is not learned attention and it is not episodic recall. It says only that the same tiny history-bearing direction can become more identity-preserving after passing through a receiver that was learned for another task.

Raw receipt: `receipts/receiver-query-smoke-20261002.json`

Summary: `receipts/receiver-query-smoke-20261002-summary.json`

## Current claim boundary

The three gates now separate three different propositions:

- **Gate 1b:** a tiny orthogonal history direction can be unusually readable by some frozen downstream heads.
- **Gate 2:** that direction is not, by itself, a useful generic cosine retrieval key.
- **Gate 3:** passing it through the frozen learned receiver does not yet produce successful retrieval, but the value-head geometry makes the real temporal signature markedly more predecessor-specific than matched controls in the three-position smoke test.
- **Same-present history:** different retained histories make a fixed ping produce different responses, but the response does not pass the declared native-continuation targeting gate on 12 cue pairs across four common positions.

The receiver-dependent hypothesis remains unproven as a useful continuation mechanism. The same-present assay removes the coarse between-position identity shortcut and separates a history-dependent response from correctly targeted continuation. Broader independent tasks and a beneficial intervention test would be needed for a stronger claim.
