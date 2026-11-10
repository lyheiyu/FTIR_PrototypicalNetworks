# Frozen heterogeneous FTIR transfer V1

Results: `results/heterogeneous_ftir_transfer_v1/`. The prior blocked-D4 audit and all earlier experiments/checkpoints stay unchanged. The new task explicitly authorizes a fixed, versioned missing-coordinate input adapter.

## Reproduce in PowerShell

```powershell
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' -m unittest discover -s tests -p test_heterogeneous_ftir_v1.py -v
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' run_heterogeneous_ftir_transfer_v1.py --smoke --output results/heterogeneous_ftir_transfer_v1_smoke_checked
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' run_heterogeneous_ftir_transfer_v1.py --validated-smoke results/heterogeneous_ftir_transfer_v1_smoke_checked
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' supplement_heterogeneous_full_mahalanobis_v1.py
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' analyze_heterogeneous_ftir_transfer_v1.py
```

The runner checks source hashes and resumes complete trials under the same manifest. Choose a new directory/version if changing the protocol. The first smoke directory records an implementation validation failure before result finalization; `smoke_checked` is the successful smoke used to authorize the full run.

## Locked inputs and data access

- Original 1868-point physical grid, 47,400-parameter M1 encoder, C4 training splits, archived PCA64 and original base covariance are reused. No optimizer, encoder training or new loss is used.
- D4: 954 unique spectra, eleven retained labels. Five exact duplicate records are removed by the existing loader. Native data cover 1761 of the locked coordinates; 107 missing coordinates are imputed with per-coordinate means from each checkpoint's archived C4 **base_train input rows only**.
- The explicit adapter uses the C4 numerical convention `native max(y)-y → airPLS → maxabs → in-range cubic → missing-coordinate mean fill`. No interpolation outside measurement bounds, mask channel, post-fill renormalization or target-statistic fitting. D4's existing loader default no-flip remains untouched. Explicit flip=True in this version is chosen to match the numerical C4 protocol, not from D4 query results and not as proof of physical transmittance/absorbance units.
- Spectrum references in D4 and C4→D4 see only common observed coordinates. Frozen PCA/M1 see the full grid with mean fill. Their representation differences are intentional and documented. All SVMs use fixed linear C=1; no D4 query tuning.
- Controlled masked C4 deletes coordinates **after original preprocessing**, queries only. Its full C4 support bank and head statistics stay fixed. This isolates coordinate sensitivity, not native truncated-interval baseline effects.

## Experiments

Three archived encoder tasks × three seeds. All external conditions use 200 repeat draws and 1/5/10/20-shot where feasible. Each D4 class has a persistent disjoint support/query pool; five query examples/class are drawn per repeat. Same-seed D4 episodes are shared across all three encoder tasks. Repeat draws are not independent trained models.

1. C4 query-only mask: same archived final-session supports/queries, four M1 embedding heads; recompute original predictions and verify against previous results, then compare masked queries.
2. C8: same final-session C4 support bank, seven methods; query C4/C8 records matched by ID and label, balanced to the minimum number of available records per class. IDs must not intersect C4 base_train or retained support IDs. Source record-ID pairing does not verify physical specimen correspondence. This is exploratory transfer between resolution-family datasets, not independent instrument evidence. Class-wise common-grid similarity auditing does not select queries or change the protocol.
3. Within D4: seven methods on fixed core9 for all four shots; expanded eligible11 at 1/5 and eligible10 at 10 shots. PA(21) and MORPHOTYPE 2(10) cannot join a fixed 20-shot panel with five disjoint queries. Strict encoder-label-disjoint results from the three `holdout_pp_ps` checkpoints are reported separately; the other six encoders have seen C4 PP/PS labels. PE is not merged into HDPE/LDPE.
4. C4→D4 incremental: six eligible checkpoints whose training base includes PP and PS. Start with original C4 PP/PS 20-shot banks, add D4 PE then PEVA (2+1+1). Base queries are D4 PP/PS; novel support and all D4 query pools are disjoint. Prototype class registration never refits prior classes. 1-NN/SVM retain all support and rebuild/refit their global classifier as appropriate.
5. Two-class anchor: same C4 PP/PS heads, complete C4, masked C4, ID-matched C8 and D4 PP/PS query conditions. C4/C8 matched records lack verified physical specimen equivalence; D4 has separate queries with the same textual class names. Spectrum references use the D4-valid interval in **all** anchor conditions.

## Outputs and interpretation

Read `results_overview_zh.md`, `transfer_comparison.md` and `d4_input_adapter_audit.md`. Required CSV files preserve episode metrics; `trial_metrics.csv` averages within each task/seed before across-trial comparison. Per-class CSV and confusion figures pool repeated query counts; they do not count unique specimens. `strict_label_disjoint_summary.csv` separates label overlap with encoder pretraining. `average_incremental_metrics.csv` averages the three sessions. Group Macro-F1 uses global one-vs-rest F1, retaining cross-group false positives.

Costs are head/incremental measurements over the first 20 repeats. Prototype registration computes the arriving class only; SVM full bank assembly/refitting is reported separately. Actual new D4 M1 feature encoding is replayed for incremental arrivals. Query prediction cost uses cached features and excludes encoder/preprocessing; within-D4 episode fit cost covers all classes in that episode. Encoder training cost is zero.

The auxiliary `head_state_bytes` field counts retained numeric bank/metric payload (plus SVM fitted arrays), not actual peak process memory: Python overhead and additional materialized support copies are excluded. It must not be interpreted as a measured RAM footprint.

`protocol_audit.json` and `frozen_encoder_verification.csv` verify hashes, frozen weights, mean fitting, unchanged PCA/prior, D4 pool separation and C8 support/training ID exclusion. Historical sources are checked against the previous manifest. Inputs, fitted adapter means/indices, feature arrays, query-pair indices and D4 split/episode files are saved for review.

All results remain exploratory. D4 physical signal units/polarity and cross-file specimen identity are unresolved. Morphotype labels are unidentified categories. Missing-coordinate mean fill does not recover measured chemical information. Base validation was previously used for checkpoint selection. Neither a numeric input shape nor good classification proves measurement equivalence. Accuracy differences between six-, nine-, eleven- and four-class protocols must not be presented as directly matched transfer gaps; the record-paired C8 and PP/PS anchor controls address narrower comparisons.

## Complete-class-covariance supplemental control

The existing full class Mahalanobis head is also reused with the previously fixed ridge .001 and original effective ridge .002, on exactly the same external features and episodes. This preserves the earlier requested complete-covariance ablation without replacing the seven requested methods. Results live in `full_mahalanobis_control/` and `full_class_mahalanobis_control_metrics.csv`; report comparisons and confusion plots are appended by the analyzer. No extra encoder inference/training or query tuning occurs. C4 full-head metrics are verified against the previous full-covariance experiment. Prototype old-class means and precisions are checked unchanged for the first repeat of every shot/stage.
