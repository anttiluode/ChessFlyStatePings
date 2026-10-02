# Same-present history review and rulings

Reviewed range: `aa698d43f617b643712b275e39f3d397445920ea` through `3fcc8a6d49b62583966b30650e956fe6f0c1289d`. One independent, read-only whole-branch review. No critical or important findings; ready to merge into its stated base.

The reviewer independently ran the full suite (116 passed, including real-artifact integration) and diff checks. A fresh complete version 2 evaluation exactly matched the committed results. Re-running version 1 matched its full runs and primary result; only the newly explicit `ping_steps` metadata differs. Summary aggregates and raw-receipt SHA-256 matched.

## Minor finding and recommendation

- Float32 subtraction of decoder outputs makes small response margins precision-sensitive. A decoder/projection-only float64 diagnostic with the recurrence, source ping and weights held fixed changed the primary native margin from about 0.0000961 to 0.0002264; mean relative policy-response error was 0.73%. Accuracy stayed 13/24 and native alignment stayed negative. **Ruling:** document numerical sensitivity, preserve the first receipts and failed gate. A reusable precision-diagnostic command is deferred; it is not necessary to support this negative result and would extend the declared assay.
- A positive synthetic end-to-end evaluator regression would strengthen a future extension. The present positive tests cover the state-specific decoder response and pair scorer; evaluator and CLI fixtures exercise unidentifiable cases. **Ruling:** retain this as a future test recommendation. The current real-artifact results were independently reproduced; no blocking coverage defect was found.

## Behaviors set aside by the reviewer

- Biological replication and empirical validity of the preprint: outside a computational code review. **Ruling:** the supplied paper was read separately for motivation, and documentation limits its connection to this artificial assay. No replication or biological mechanism is claimed.
- Beneficial recurrence intervention and chess-strength improvement: outside this readout measurement. **Ruling:** the gate explicitly scores an untouched internal next transition, and the probe response never enters recurrence. A behavioral intervention would require a separate experiment.
- CUDA numerical parity: no CUDA validation performed. **Ruling:** report CPU execution and exact artifact hashes; numerical/device differences remain an explicit limit. CPU tests and deterministic reruns support the reported CPU outcome.
- Earlier gates and deployed hosting: outside the reviewed implementation changes. **Ruling:** baseline APIs remain unchanged and their tests pass. The site source was updated; no deployment or merge of earlier stacked branches is claimed.

## Source timing decision

The original version 1 source was structurally zero because its first readout was zero. **Ruling:** correcting the causal source timing was necessary to exercise the authorized question. The first receipt and source-only diagnostic were preserved, and the version 2 amendment was committed before its measurement. No case, magnitude, threshold or control was tuned. This decision is disclosed in the analysis and final report.
