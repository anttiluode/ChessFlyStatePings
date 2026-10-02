# Gate 4 — Listener Replacement

## Question

Gate 3 held the frozen ChessFly receiver fixed and replaced the temporal direction. Gate 4 reverses that control axis: it holds the **same real temporal signature fixed** and replaces only the downstream listener.

For each position, step 4 supplies the memory record and step 5 the query record. The association-space temporal signature is the same sign-symmetric central difference used by Gate 3:

```text
g(a) = (A(h + a r_perp) - A(h - a r_perp)) / (2a)
```

The identical `g(a)` is then compared in association space and after two frozen learned listeners: the value head and policy head. Each learned head is compared with 32 column-permuted versions of its own weight matrix. Column permutation preserves the listener's singular values exactly but changes which learned association coordinate each weight hears. Value and policy control `i` use the same coordinate permutation.

## CUDA smoke result

Configuration: 3 positions, rho=0.75, seed=0, magnitudes 1/2/4, 32 listener controls, zero instabilities.

Raw history remains a failed direct address: top-1 0/3, MRR 0.389, margin -0.02204. Association-space signatures remain 1/3 top-1 at all magnitudes.

### Value listener

| magnitude | real margin | shuffled-listener mean | margin percentile | real MRR | MRR percentile |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.00589 | -0.08084 | 68.2% | 0.667 | 83.3% |
| 2 | -0.00408 | -0.09609 | 77.3% | 0.667 | 84.8% |
| 4 | -0.00163 | -0.10532 | 77.3% | 0.667 | 84.8% |

The learned value listener is consistently more predecessor-specific than spectrum-preserving shuffled versions of itself. The effect strengthens in raw margin gap as magnitude grows, but top-1 remains only 1/3 and every real margin remains negative.

### Policy listener

| magnitude | real margin | shuffled-listener mean | margin percentile | real MRR | MRR percentile |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.05622 | -0.03372 | 34.8% | 0.611 | 40.9% |
| 2 | -0.08752 | -0.05182 | 22.7% | 0.611 | 56.1% |
| 4 | -0.10600 | -0.06026 | 16.7% | 0.611 | 57.6% |

The learned policy listener shows the opposite margin pattern: the intact learned coordinate organization is worse for predecessor matching than its shuffled-listener controls, increasingly so as magnitude grows.

## Interpretation

Gate 4 is not a successful retrieval result. Neither learned listener retrieves the correct predecessor reliably on this three-position smoke set.

What it does show, provisionally, is a **listener-specific alignment effect**. The same preserved temporal signature becomes relatively predecessor-specific under the learned value geometry and relatively anti-specific under the learned policy geometry, compared with spectrum-preserving replacements of each listener.

That sharpens the relational hypothesis:

```text
meaning/address-like structure is not solely in the ping
and is not solely in the listener;
it can live in their learned alignment.
```

This is compatible with the observer-replacement direction suggested by `InsideTheWave`, but it does not establish biological memory, attention, consciousness, or quantum-like physics. The smoke set is n=3; larger held-out sets are needed before treating the percentile asymmetry as general.

Raw receipt: `receipts/listener-replacement-smoke-20261002.json`

Summary: `receipts/listener-replacement-smoke-20261002-summary.json`
