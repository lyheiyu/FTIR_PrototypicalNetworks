# C4 frozen M1: full class Mahalanobis ablation V1

This supplements `results/external_transfer_prototype_v1`. It does not modify that experiment, its scripts, checkpoints, preprocessing or sessions.

## What “full” means

All three Mahalanobis variants use full 128 × 128 matrices, including off-diagonal entries. The new control is **class-specific empirical covariance without shrinkage toward a base prior**. This is different from a diagonal covariance control.

For a class with n support embeddings, μ is their mean and C = Σ(z−μ)(z−μ)ᵀ/n. Classification minimizes (z−μ)ᵀΣ⁻¹(z−μ), without a determinant term or class prior:

| Method | Σ | Data used to estimate metric |
|---|---|---|
| Class full, ridge .001 | C + .001 I | Support of that class only |
| Class full, legacy .002 | C + .002 I | Support of that class only |
| Shared full | B + .001 I | Archived base-training within-class covariance B |
| Class shrinkage | α C + (1−α) B + .001 I | Class support and archived base-training prior |

α = (n−1)/(n−1+5), fixed before this ablation. Both shared and shrinkage have additional access to the base-training prior; this difference is explicit. No method fits on novel queries or future classes. Euclidean, M1 1-NN and M1 SVM reference metrics are reused unchanged from the completed parent study.

The original `FTIR_fewShot_Learning_adapted_M_distance_MCCV2.py`, functions `compute_prototypes_and_covariances` and `mahalanobis_distance`, each add .001 I by default. Thus .002 reproduces its covariance formula. This ablation uses cached float64 features and omits the monotone square root for classification. It is formula-equivalent, **not a bitwise reproduction of the original float32 Torch implementation**, and does not rerun that script’s training or augmentation. The .001 control isolates covariance shrinkage at matched ridge. These values were selected from existing code, not tuned against these query results.

## Singular covariance

rank(C) ≤ n−1. Dimension is 128 and every class here has at most 20 support embeddings. Thus every empirical class covariance is singular before ridge; an ordinary zero-ridge inverse is undefined. This study does not silently substitute a pseudoinverse. At one shot C=0; the class metric becomes I/ridge. Base classes retain 20 samples, so their metrics still differ from the arriving one-shot class metric.

## Protocol and commands

Three archived tasks × three seeds × 1/5/10/20 shots × 200 matched episodes × sessions 0/1/2. Four base classes retain the same 20 exemplars per class. Saved support/query indices and archived frozen M1 embeddings are reused exactly. No encoder training occurs. Shared and shrinkage metrics are freshly recomputed and checked against the parent for every context. Built-in checks compare the new implementation with `Head('class')`, verify the original double-ridge formula, the one-shot case and preservation of previous class state on registration.

PowerShell:

```powershell
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' run_full_mahalanobis_ablation_v1.py --smoke --output results/mahalanobis_full_ablation_v1_smoke
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' run_full_mahalanobis_ablation_v1.py --validated-smoke results/mahalanobis_full_ablation_v1_smoke
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' analyze_full_mahalanobis_ablation_v1.py
```

The runner resumes completed trials with an identical manifest; it rejects changed source/configuration. Incomplete trials are recomputed. Results are written under `results/mahalanobis_full_ablation_v1`.

## Metrics, costs and interpretation

All sessions report Accuracy and Macro-F1 for base, previous novel, current novel, both introduced novel and all seen classes. Group Macro-F1 averages global one-vs-rest class F1, retaining false positives across groups. Per-class precision/recall/F1 and confusion counts are saved. Final 5/10-shot confusion plots show all six named classes separately for each task and metric.

The primary paired summary first averages repeats and the 5/10-shot conditions within each task/seed, then compares nine paired trial values. Episodes within a task/seed are not treated as independent trained-model replications. No query-tuned hyperparameters or confirmatory significance claims are used.

Costs are freshly measured for all four methods in the same process, one numerical thread, first 20 repeats. Registration computes only the arriving class mean/covariance/inverse; old states stay fixed. Prediction uses cached query embeddings. Encoder inference, input preprocessing, Python object overhead and encoder weights are excluded from these head costs. Initial base registration is outside incremental registration timing. All dense metric matrices, global priors, means and counts are included in numeric head-state memory. Encoder training cost for this supplement is zero.

The new class-full head caches its inverse at registration, as the shrinkage head does. The original Torch script instead inverts a covariance inside the query/class distance loop. Consequently these timings compare efficient implementations of the metric formulas; they are not runtime measurements of that original script.

The full-class control requires no global covariance prior. At six classes its head state is 792,624 bytes; shrinkage retains the additional 131,072-byte base prior, totaling 923,696 bytes. Both retain six full per-class precision matrices. Strong accuracy does not imply a smaller memory footprint than a 1-NN feature bank.

This is exploratory C4 analysis: novel queries have already been inspected; base queries reuse validation data previously involved in checkpoint/C selection; specimen grouping remains unknown. It adds no independent external transfer evidence. D4 evaluation remains blocked by the previously audited input-grid incompatibility.

Whether class-specific adaptive Mahalanobis improves over Euclidean/shared and whether shrinkage improves over the unshrunk class-specific method are **two separate comparisons**. Neither should be inferred from only the shared-versus-shrinkage result. The report must retain any shot-dependent failures and task/seed exceptions.
