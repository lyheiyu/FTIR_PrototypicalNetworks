# Protocol and leakage audit: final C4/C8 round

| dataset | native_points | native_min | native_max | median_spacing | minimum_spacing | maximum_spacing | locked_grid_points | locked_min | locked_max | extrapolated_coordinates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c4 | 3736 | 399.193 | 4000.604 | 0.964 | 0.964 | 0.964 | 1868 | 399.193 | 4000.604 | 0 |
| c8 | 1869 | 399.193 | 4001.569 | 1.928 | 1.928 | 1.929 | 1868 | 399.193 | 4000.604 | 0 |

Both measured native axes cover the exact locked 1868-point historical physical grid. Cubic interpolation is bounded; no new axis, extrapolation, padding or D4 input is used. The native coordinate arrays and locked grid are saved exactly.

The legacy pipeline applies airPLS on native sample-index grids BEFORE interpolation. Identical numerical settings at different native point densities do not establish identical baseline smoothing in physical wavenumber units. The observed contrast therefore includes native sampling and legacy preprocessing effects, not a pure causal optical-resolution intervention. Signal absorbance/transmittance semantics and acquisition hardware remain unverified.

## Conservative record grouping

| label | record_groups | max_original_ID_component |
| --- | --- | --- |
| HDPE | 495 | 2 |
| LDPE | 299 | 3 |
| PET | 494 | 3 |
| PP | 497 | 3 |
| PS | 61 | 36 |
| PVC | 201 | 4 |

Original source IDs are blocked across C4/C8; native duplicate aliases from BOTH 3000-row mapping files are joined transitively, then exact common-grid processed equality is also blocked. This yields 2047 shared groups, including only 61 PS groups. Every selected representative has the same retained original ID in both resolutions. One representative/group/resolution is used for training and evaluation, preventing sample-count and duplicate-group-weight differences between sources. ID-based grouping is a conservative blocking proxy, not verified physical specimen provenance.

Conservative transitive grouping may merge genuinely different objects, and one representative omits within-component spectral variation. The tested population is this fixed paired representative set, not every native unique spectrum. Original source-file hashes and physical axes are checked against the retained deduplication provenance.

## Why both encoders were trained afresh

Historical C4 base rows were all used for training or checkpoint validation, leaving no clean base-query pool under the new requirement. Therefore both C4 and C8 M1 were retrained from scratch on common group splits, with unchanged architecture, Adam .001, CE, batch64 and 60 epochs. Historical checkpoints are unchanged. Each pair uses identical initialization seed, epoch group orders and update count; only source spectral inputs differ. Selection uses the lowest SOURCE base-validation loss; no early stopping or target/evaluation label selection. Temporary classification heads are not included in frozen encoders.

Base groups: query=max(15,floor(.20n)); validation=max(5,floor(.15n)); remaining groups train. Base references are fixed 20/group/class sampled from train. Novel groups are split 50/50 support/query, allowing 1/5/10/20-shot and fifteen disjoint query groups/class even for PS. Roles are identical across resolutions, so target query counterparts also cannot enter source training or validation. Within every episode, query groups and support groups are disjoint. Queries and maximal support order are shared across all settings, classifiers and nested shots.

## Models and fixed heads

M1: original position-preserving multiscale CNN, 47,400 encoder parameters and 128-dimensional ReLU embedding. Independent full class covariance uses residual.T@residual/n + .001 I, cached inverse, quadratic distance, no determinant or class prior. Shrinkage uses alpha=(n−1)/(n−1+5), same .001 ridge and the encoder SOURCE base_train within-class prior; E/F do not refit it on target references. All empirical support covariances are singular before ridge because n≤20. The original script’s effective double ridge .002 is a historical formula, not this fixed matched-ridge main control.

PCA64 is source-base-train-only, randomized SVD, fixed seed, no whitening; its own 64×64 source base prior is frozen. It uses the same nominal ridge .001 and tau5 as M1-128. Different dimensions and variances mean relative regularisation is NOT matched; source-only scale diagnostics are saved in regularisation_geometry.csv. Raw SVM is linear C=1; all baselines retain exactly twenty base exemplars plus K per introduced novel class, not all source training rows.

Prototype registration changes arriving-class statistics only. Existing class means/precision arrays are checked unchanged for the first repeat of every shot/session/setting. The evaluation receives frozen feature arrays and no optimizer/encoder; checkpoint file hashes are verified after evaluation. Raw A=F and B=E metric invariance provides sanity controls for matched target-support encoder-source comparisons.

## What is and is not independent

The protocol excludes known exact duplicates, linked record IDs and their transitive aliases across roles and resolutions. No supplied metadata establishes specimen-level identity, so no result is certified specimen-independent. C4/C8 are resolution-family datasets, not independent instruments; matching record IDs is not proof of identical physical acquisition. Common-grid correlation diagnostics are descriptive only and never select queries.

Both source datasets and earlier results were already inspected. The new query groups are excluded from this round’s training/model selection, but are NOT a genuinely untouched confirmatory holdout. All results remain exploratory; nine task/seed pairs also share datasets, and 200 repeats are not independent specimens or 200 independent trained models.

Historical H (old C4 encoder, C4 base AND novel support, C8 query) is retained separately. It is not setting C, which registers novel C8 support, and uses different checkpoints/group partitions. Do not attribute H-versus-C differences solely to novel-support source.

Audit passed: 18 encoders, 86400 raw invariance checks, 8424 old-class state checks; zero incremental encoder updates.
