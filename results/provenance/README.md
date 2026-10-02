# Receipt code history

The [local history bundle](same-present-history-local.bundle) preserves the exact original commits referenced by the same-present history receipts, including the design and source-timing amendment committed before their respective runs. It contains source code, tests, documentation and receipts, including the stacked base history. It contains no upstream checkpoint, graph binaries or supplied paper PDF.

GitHub's connected Git-data API creates a new publication commit. The measured code commits are retained here byte for byte, even though the publication commit has different Git metadata. Receipt files are unchanged.

From a clone of this branch:

```bash
git bundle verify results/provenance/same-present-history-local.bundle
git fetch results/provenance/same-present-history-local.bundle HEAD:refs/remotes/receipts/local-history
git show e269343c14b0572ecfb868335ca23e6727d598a3:src/chessfly_statepings/history_lens.py
```

Version 1 ran at `555a6211eb82e9a310b437974fa34dc9ba124949`. Version 2 ran at `e269343c14b0572ecfb868335ca23e6727d598a3`. The bundle's final local commit is `9c64134de26889872fd48b9977427137558e8c55`, after independent review and documentation of numerical limits. The publication tree keeps the reviewed code and raw receipts identical and adds this provenance record.
