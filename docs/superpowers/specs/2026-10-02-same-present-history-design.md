# Same present, retained history, fixed delayed ping

## Question and authorization

Antti requested continued implementation of: **With the same present input, do different retained histories make the same delayed ping produce different, correctly targeted continuations?** Earlier session authorization explicitly preapproves design choices and native implementation. This document fixes the protocol before measurements; no further design approval is requested.

## What “correct” means here

The target is the frozen model's **next unmodified settling transition**, withheld from ping construction. This is an internal one-step continuation assay, not chess optimality, episodic recall, or a trained forecast task. It can answer whether a common probe elicits a state-specific direction that points toward the appropriate continuation. A large response or a different response alone is insufficient.

## Protocol

Each case supplies a present FEN and two distinct cue FENs. The two histories use the identical cue multiset in opposite order: A,B and B,A at settling steps 1 and 2. Both receive exactly the same encoded present input from step 3 onward. These are controlled sensory snapshots within one settling run, not legal game trajectories or state carried across chess moves. All weights, graph edges, scales, and shifts stay frozen.

The query arrives after 1 or 2 shared-present steps (global steps 3 or 4). The held-out target is the next unmodified recurrent step (4 or 5). A third, present-only trajectory supplies the erase-history control and a fixed ping constructed **only from its first two steps**, using the existing orthogonal fast-minus-slow coordinate, rho=0.75. The ping is identical for both histories, has no access to either history-specific target, and is not reprojected against individual receiver states.

For decoder D, receiver state h_i and fixed ping p, measure the balanced odd response g_i(a)=[D(h_i+a p)-D(h_i-a p)]/(2a). Target delta_i=D(h_(i,next))-D(h_i). Store the complete 2x2 response/target cosine matrix. Center value logits and legal policy logits to remove common-logit offsets. Association space is secondary; **centered legal policy** is primary. Querying reads the receiver at the arrival time; it does not restore a frozen historical state.

## Controls and identification

- Erased history: present input at all steps; two identical query states, scored against the original two targets.
- Swapped history: exchange the two retained responses while keeping target labels fixed; no target recomputation.
- State-blind decoder: the affine decoder response W p and its frozen heads, independent of receiver state.
- 32 shuffled, re-orthogonalized, norm-matched pings: each control remains identical across both histories. Seed 0. One control permutation per case is shared across delays and magnitudes.
- Record hidden/readout separation, response separation, ping norm, target norms and target-direction separation. Zero or indistinguishable targets are marked unidentifiable; finite raw matrices remain available. Treat cosine ties within 1e-6 as half credit, never arbitrary argmax success.
- Identical cue encodings, malformed/terminal FENs, duplicate case IDs, nonfinite parameters and out-of-range delays fail explicitly. No future-dependent ping selection, magnitude selection or per-case gain fitting.

## Fixed evaluation

Default data: 12 prespecified opening cases with equal-material cue alternatives. Default delays 1,2 and magnitudes 0.5,1,2. Primary setting: delay 1, magnitude 1. Other settings are sensitivity checks. At least 12 identifiable primary cases are required for a gate decision. Success requires mean native margin >1e-6, tie-aware accuracy >=0.75, positive mean native alignment, margin improvement >1e-6 over erase-history and at least the 95th empirical percentile among the 32 matched-ping controls. Fewer identifiable cases means inconclusive; a missing primary setting means not run. Failure remains a first-class result.

The gate does not isolate history information beyond the complete present hidden state; that hidden state is the retained history carrier being tested. It does not implement an axonal waveform, cortical tuft, Martinotti cell, oscillatory phase, PAC, consciousness, or quantum mechanism.

## Source motivation

Martin-Burgos et al., *Action potential waveforms are state-dependent*, bioRxiv, version posted 21 September 2026, doi:10.64898/2026.09.15.751814 (supplied PDF; not peer reviewed). The paper reports within-cell waveform relationships with input drive and heterogeneous relationships with local-field-potential amplitude/variability. Its discussion cites earlier evidence that presynaptic waveform width can affect transmission. Its new experiments do not establish useful decoding of transmitted historical information by a receiver. This gate probes a computational receiver-side question in an artificial model; it is not a replication. Cite the paper; do not commit the supplied PDF.

## Version 2 amendment: source timing, before its measurement

The preserved version 1 receipt (`results/receipts/history-lens-20261002.json`, code `555a621`) is inconclusive: zero identifiable cases in every geometry and setting. A source-only diagnostic found that the readout at step 1 is exactly zero for all four declared common inputs. Consequently the two-step residue is exactly rho times the step-2 readout in real arithmetic; orthogonalization must remove it. The measured ping norms, 0 to 1.33e-7, are rounding residue. This is a source-construction limitation, not evidence against retained history.

Version 2 makes one causal timing correction. Use the present-only reference's first **three** steps, then deliver that unchanged ping at global step **4**, one step after construction. Both receivers have now had **two** identical-present steps; global step **5** remains the unmodified held-out target. With only five model steps available, version 2 has one legal delay, 2, and retains all three magnitudes, the same 12 cases, rho, seeds, controls, score definitions, and thresholds. Its fixed primary is delay **2**, magnitude **1**. This choice uses the absence of source activity and the earliest available noncollinear prefix; it does not select on response/target alignment.

Expose `--ping-steps 3` as the new default and retain `--ping-steps 2 --delays 1 2` for version 1 reproduction. Reject any query that arrives before or at the ping construction step. Label receipts `same-present-history-lens-v2` or `v1` and record ping construction, query and target steps explicitly. Add a regression with a zero first readout and independent second/third readouts, and causal tests that changing steps after the selected prefix cannot change the ping. Preserve both first receipts, regardless of the version 2 result.
