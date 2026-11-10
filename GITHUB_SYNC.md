# GitHub synchronization scope

This repository includes research source code, experiment instructions, configurations,
tests, manuscript drafts, and selected experiment reports, summary tables and figures.

Raw datasets, trained model weights, feature caches, detailed per-episode predictions,
and smoke/check outputs remain local. Their exclusion is intentional; the GitHub copy
is not a complete backup of experimental artifacts. Source snapshots and provenance
files embedded in local result directories are also not included. Reproducing runs
requires the datasets and environment described in the experiment run instructions.

Useful report entry points:

- `results/harmonization_v2_dev_c4/RESULTS_CONCLUSIONS.md`: interpolation/grid comparison.
- `results/protonet_bidirectional_final/summary_zh.md`: final ProtoNet comparison.
- `results/bidirectional_c4_c8_final/final_results_zh.md`: bidirectional transfer.
- `results/mahalanobis_full_ablation_v1/conclusion.md`: full covariance comparison.

For subsequent updates, inspect `git status --short`, then run:

```powershell
git add --all
git diff --cached --stat
git diff --cached --check
git commit -m "Update research code and reports"
git push origin main
```

Review newly added files before committing. Ignore rules do not apply to files that
Git already tracks.
