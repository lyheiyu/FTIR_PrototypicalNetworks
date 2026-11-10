# Final C4/C8 bidirectional experiment

Results: `results/bidirectional_c4_c8_final/`. D4 is excluded. This is the final fixed experimental round before manuscript writing; no architecture, loss, augmentation or hyperparameter search is introduced.

## Reproduce

```powershell
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' -m unittest discover -s tests -p test_bidirectional_final.py -v
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' run_bidirectional_c4_c8_final.py --smoke --output results/bidirectional_c4_c8_final_smoke
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' run_bidirectional_c4_c8_final.py --validated-smoke results/bidirectional_c4_c8_final_smoke
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' analyze_bidirectional_c4_c8_final.py
```

With identical source/configuration hashes, complete training/evaluation trials are reused. Incomplete source training is restarted from its fixed initialization; a complete source model is reused if evaluation was interrupted. Changed code or configuration requires a new versioned output directory and new smoke validation. Historical models are never overwritten.

## Important protocol decisions

The old C4 experiment used every base row for training or validation, so its checkpoint has no base-query pool untouched by checkpoint selection. Reusing it would violate the new task's query-selection requirement. Therefore **both source M1 models are trained afresh**, using one common record-group partition. This changes only data partition/representation selection, not the original M1 architecture, loss or optimizer protocol. All old checkpoints remain separate historical results.

Conservative components join shared original IDs across C4/C8, both files' exact-native duplicate aliases, and any exact processed-grid equality. There are 2047 paired components: PET494, HDPE495, PVC201, LDPE299, PP497, PS61. Every selected component has a shared retained original ID, so the same record ID is used in both resolutions. Only one representative per component/resolution enters training/evaluation, equalizing group counts and update budgets.

These are **record/duplicate components, not verified physical specimens**. Even a matching ID does not establish identical acquisition. All earlier datasets/results have been inspected; fresh model-specific holdout exclusion does not create a genuinely untouched confirmatory dataset. No claim of independent-instrument or specimen-independent generalization is supported.

Conservative transitive blocking can overmerge genuinely distinct objects, while selecting one paired representative omits within-component spectral variation. Conclusions apply to the fixed paired representative set; it is not equivalent to evaluating all independently deduplicated native spectra.

The original locked 1868 physical coordinates are measured by both native axes; no extrapolation or new grid is needed. Original numerical preprocessing stays `native max-y → native airPLS → native maxabs → cubic to locked grid`. The index-based airPLS smoothing precedes interpolation, so fixed parameters do not guarantee equal physical smoothing at different native densities. Signal units and acquisition metadata remain unresolved.

## Matched data and training

Three original 4+1+1 tasks and three seeds. Base groups: max(15, floor20%) query, max(5, floor15%) validation, remainder train. Novel groups split 50/50 support/query. For base PS this gives 37 train, 9 validation, 15 query groups. All source/query/reference roles apply to both resolutions jointly.

Each C4/C8 model pair uses the same initial seed, exact group epoch orders, 60 epochs, batch64, Adam .001 and four-class cross entropy. Both source models have 47,400 encoder parameters. Selection is lowest source-base-validation loss, with no early stopping or target-query tuning. The temporary linear head is absent from the saved encoder used for evaluation. Fit PCA64 and the separate PCA/M1 source base covariances only on source base_train.

After training, encode both sources with each frozen source encoder. Incremental evaluation only receives saved features and no optimizer. Hold 20 base groups/class; introduce K=1/5/10/20 novel supports/class, with 5/10 primary. Each of 200 repeats uses the same fifteen query groups/class across all six settings; query groups remain fixed when later classes arrive, and support prefixes are nested across K.

## Six settings and interpretation

| Setting | Encoder | Base references | Novel support | Query |
|---|---|---|---|---|
| A | C4 | C4 | C4 | C4 |
| B | C8 | C8 | C8 | C8 |
| C | C4 | C4 | C8 | C8 |
| D | C8 | C8 | C4 | C4 |
| E | C4 | C8 | C8 | C8 |
| F | C8 | C4 | C4 | C4 |

Source-specific shrinkage priors stay fixed even in E/F. C−A and D−B are directional changes relative to each source's within-source reference. C−D is the absolute direction contrast; (C−A)−(D−B) compares relative degradation. E−C and F−D isolate base-reference source with the encoder/prior fixed. E−B and F−A match target supports/queries but vary representation source; shrinkage also changes source prior, while M1 1-NN and independent full covariance use no global prior. Raw A=F and B=E metrics must be exactly invariant. E/F require twenty labelled target base references/class, so their bank-alignment benefit is not free of additional target-reference data access.

Historical H uses legacy C4 encoder, C4 base **and C4 novel support**, C8 query. It is retained in a separate historical CSV; it is not C and uses different checkpoints/splits, so H-versus-C does not isolate support source.

## Fixed classifiers

Raw spectrum 1-NN; linear spectrum SVM C=1; PCA64+shrinkage; M1+1-NN; M1+independent full class covariance; M1+shrinkage. Every class-specific metric uses residual covariance divisor n, ridge .001 and quadratic nearest-prototype scores without determinant/class priors. Shrinkage alpha=(n−1)/(n−1+5); its prior is the original source-base class-residual covariance. Existing old class means/precision matrices do not change when a new class registers.

Full covariance means a dense 128×128 covariance with ridge, not an invertible unregularized empirical estimate: support counts ≤20 make the latter singular. The legacy script's effective double .002 ridge is documented historically; main full/shrinkage controls here both use .001, fixed before queries. PCA is 64-dimensional, unwhitened, source-only. Equal nominal ridge does not imply equal relative regularization or capacity versus M1-128; source-only geometry diagnostics are included.

## Read outputs

Start with `final_results_zh.md`, `manuscript_results_en.md` and `protocol_and_leakage_audit.md`. Required episode metrics, paired comparisons, per-class statistics and `confusion_matrices/` are saved. Primary summaries average repeats and 5/10 shots within task/seed before summarizing nine pairs. The nine pairs share source datasets and repeated episodes are not independent specimens; win counts are descriptive, not confirmatory significance tests.

The main Macro-F1 averages per-episode global one-vs-rest F1; per-class CSVs derive precision/recall/F1 from cumulative repeated-query confusion matrices. These are distinct aggregations. Old-novel retention uses the same first-novel queries before/after the second class arrives. Boundary competition can reduce accuracy despite unchanged encoder and old-class states.

`reverse_training_summary.csv` includes both C4 and C8 training for audit, not just reverse models. `frozen_encoder_verification.csv`, `protocol_audit.json`, source snapshots, split/group/alias mappings, exact physical axes, initialization/order hashes and fixed-source PCA/prior arrays make the conditions reviewable. Extra head timing uses cached features and excludes encoder/preprocessing; it is not end-to-end latency.

Base prototype template initialization is outside the timed incremental updates. Session-0 prototype registration is a no-op; compare arrival sessions 1/2 for local registration. `manuscript_results_en.md` is the concise Results draft; `manuscript_results_detailed_en.md` retains the longer numerical description.
