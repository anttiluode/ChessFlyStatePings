# ChessFlyStatePings results

This directory keeps small, inspectable receipts from real-artifact runs. The model checkpoint and fly-derived graph stay frozen; receipts record the tested intervention, exact inputs, versions, artifact hashes, and measured outcomes.

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

The negative result is specific to this memory bank, this orthogonal coordinate, equal-weight cosine scoring, rho=0.75, and three smoke positions. It does not test learned attention, a learned associative reader, cross-move biological memory, or consciousness.

Raw receipt: `receipts/query-memory-smoke-20261001.json`

Summary: `receipts/query-memory-smoke-20261001-summary.json`

## Current claim boundary

What survives so far is narrower and more interesting than the original literal story: the settling trajectory leaves a small direction that some trained downstream geometry reads unusually strongly, but the first explicit "ping queries recorded history" implementation is falsified by its smoke test. Any stronger query mechanism now needs a reader or geometry that is actually learned for retrieval rather than imposed after the fact.
