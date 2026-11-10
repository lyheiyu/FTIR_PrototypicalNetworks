# Manuscript-ready Results: episodic training ablation

We compared cross-entropy (CE) and episodic prototypical training using the same 47,400-parameter position-preserving M1 encoder, 128-dimensional embeddings, locked 1,868-coordinate physical grid, grouped partitions, and initialization seeds. Each of three class-holdout tasks was repeated with three seeds for both C4 and C8. The two held-out classes were excluded from encoder training and source validation. Episodic training used four-way episodes with five support and ten query spectra per class, squared Euclidean prototype logits, and query cross-entropy. Adam (.001) update counts were matched to the 60-epoch CE reference (780, 960, and 840 updates across tasks); this matches optimization steps rather than example exposures or the number of directly supervised query labels. Checkpoints were selected by mean loss on 20 fixed source-validation episodes (five support and four query spectra per class), without incremental query access. No CE encoder was retrained.

All encoders were frozen before 4+1+1 registration and evaluated using identical class arrivals and support/query episodes. We evaluated one, five, ten, and twenty novel-class support spectra, with five and ten shots prespecified as primary. Base classes retained twenty references each. Settings A/B denote within-resolution C4/C8 evaluation; C/D use source-resolution base references and target-resolution novel support/query; E/F additionally replace the base references with labeled target-resolution data. Independent full-covariance and class-specific shrinkage heads used the existing ridge .001, covariance divisor n, and shrinkage coefficient (n−1)/(n−1+5). Shrinkage priors were estimated exclusively from the corresponding encoder's source base-training embeddings and frozen thereafter. Neither old class states nor encoders were refitted during incremental sessions.

The principal negative result is that ordinary episodic M1 with Euclidean prototypes did not improve mean novel-class accuracy in any of the six settings: the paired differences were A -4.83, B -6.02, C -7.74, D -2.91, E -9.18, F -9.85 percentage points. Its within-resolution overall gains (A +8.05, B +8.98) therefore should not be described as improved novel-class generalization; they mainly reflect higher base-class accuracy. Episodic training with the shrinkage head improved overall source-bank cross-resolution accuracy over CE with the same head by +3.16 (C) and +3.20 (D) percentage points, with positive overall differences in seven of nine task-seed pairs in each direction. However, episodic shrinkage remained below the fixed PCA64 shrinkage reference in mean overall accuracy in every setting, including C -8.85 and D -9.63 percentage points.

For the episodic encoder in C/D, shrinkage increased mean novel-class accuracy over full Mahalanobis by 12.86/10.73 percentage points but changed overall accuracy by -0.59/-1.96. This discrepancy reflects the base-versus-novel tradeoff: shrinkage base-to-novel error rates reached 13.98%/19.12%, versus 2.20%/3.51% with full Mahalanobis. Thus improved novel recall alone does not establish a better incremental classifier. Target-aligned base references recovered 14.00/17.44 overall points for episodic shrinkage in E−C/F−D, but required additional labeled target base data.

Feature scale changed substantially (mean training embedding norms: CE C4/C8 110.79/105.58 versus episodic 10.52/8.42). The fixed .001 ridge therefore imposes much stronger relative regularization in episodic embedding space. These results characterize the locked training-and-head protocols, not a scale-invariant or purely loss-form comparison. No rescaling or retuning was performed after observing these findings.

With the Euclidean head held fixed, episodic minus CE final overall accuracy differences were A +8.05, B +8.98, C -5.86, D +2.00, E +4.31, F -0.88 percentage points (settings A–F), and combined novel-class accuracy differences were A -4.83, B -6.02, C -7.74, D -2.91, E -9.18, F -9.85. With the full Mahalanobis and shrinkage heads held fixed, the overall differences were A +1.34, B +0.49, C -0.91, D -2.55, E -0.86, F +0.69 and A +1.38, B +0.18, C +3.16, D +3.20, E -0.03, F +1.24, respectively. These paired comparisons contrast episodic and CE training at a fixed inference rule. The contrast includes class-balanced episodic sampling versus row-wise CE training and different source-validation objectives; it does not separately identify a pure loss-form effect. In contrast, comparing shrinkage and full covariance on the same episodically trained encoder yielded A +4.30, B +4.62, C -0.59, D -1.96, E +4.39, F +4.61 percentage points, while shrinkage versus Euclidean yielded A +7.65, B +8.49, C +8.22, D +9.80, E +13.01, F +18.72; these are metric-head effects and must not be attributed to episodic training.

The final-session primary accuracy and novel-class accuracy are reported below. The PCA64 shrinkage reference remained fixed and source-trained. Episodic M1 with shrinkage minus PCA64 with shrinkage yielded A -0.74, B -1.82, C -8.85, D -9.63, E -2.58, F -0.97 percentage points. The source-to-target accuracy gaps and their objective interactions are reported separately to distinguish target recognition from changes in the within-source reference; a smaller gap alone does not establish higher target accuracy. Target-aligned reference banks require additional labeled target base-class data and therefore represent a different data-access condition.

| setting | CE_euclidean | EP_euclidean | CE_full | EP_full | CE_shrinkage | EP_shrinkage | PCA64_shrinkage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 80.825 | 88.872 | 90.892 | 92.227 | 95.149 | 96.525 | 97.265 |
| B | 78.557 | 87.541 | 90.922 | 91.410 | 95.844 | 96.027 | 97.845 |
| C | 78.047 | 72.185 | 81.902 | 80.992 | 77.246 | 80.403 | 89.251 |
| D | 66.683 | 68.679 | 82.988 | 80.442 | 75.282 | 78.481 | 88.112 |
| E | 77.079 | 81.389 | 90.879 | 90.015 | 94.428 | 94.403 | 96.981 |
| F | 78.080 | 77.197 | 90.617 | 91.303 | 94.678 | 95.916 | 96.883 |


| setting | CE_euclidean | EP_euclidean | CE_full | EP_full | CE_shrinkage | EP_shrinkage | PCA64_shrinkage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 74.225 | 69.394 | 76.726 | 77.382 | 90.860 | 91.346 | 93.072 |
| B | 75.428 | 69.411 | 77.795 | 78.861 | 94.118 | 93.430 | 96.777 |
| C | 75.147 | 67.409 | 81.713 | 82.588 | 94.587 | 95.444 | 96.428 |
| D | 74.322 | 71.411 | 77.907 | 81.404 | 91.314 | 92.137 | 93.404 |
| E | 75.570 | 66.395 | 78.265 | 77.744 | 92.596 | 91.283 | 95.505 |
| F | 73.228 | 63.380 | 76.838 | 76.911 | 89.048 | 89.759 | 92.100 |


For shrinkage and the fixed PCA reference, the base-class and previously/currently introduced novel-class accuracy, base-to-novel error rate (%), and first novel-class change between sessions (percentage points) were as follows:

| setting | method | base_accuracy | previous_novel_accuracy | current_novel_accuracy | base_to_novel_error_rate | previous_novel_change_pp |
| --- | --- | --- | --- | --- | --- | --- |
| A | CE_shrinkage | 97.293 | 90.272 | 91.448 | 2.478 | -9.256 |
| A | EP_shrinkage | 99.115 | 91.276 | 91.417 | 0.862 | -6.341 |
| A | PCA64_shrinkage | 99.361 | 94.502 | 91.643 | 0.465 | -4.256 |
| B | CE_shrinkage | 96.707 | 92.774 | 95.461 | 2.479 | -5.789 |
| B | EP_shrinkage | 97.326 | 92.854 | 94.006 | 1.260 | -3.415 |
| B | PCA64_shrinkage | 98.379 | 95.728 | 97.826 | 0.767 | -2.578 |
| C | CE_shrinkage | 68.575 | 91.909 | 97.265 | 18.143 | -7.135 |
| C | EP_shrinkage | 72.883 | 95.067 | 95.820 | 13.984 | -3.431 |
| C | PCA64_shrinkage | 85.662 | 95.783 | 97.072 | 7.822 | -3.372 |
| D | CE_shrinkage | 67.267 | 91.181 | 91.446 | 23.266 | -8.420 |
| D | EP_shrinkage | 71.652 | 92.306 | 91.969 | 19.124 | -5.946 |
| D | PCA64_shrinkage | 85.466 | 94.507 | 92.300 | 7.519 | -4.909 |


Bidirectional HDPE/LDPE and PP/PS confusion rates (%) below average the two directional errors only in the corresponding genuine novel-pair holdout task, rather than mixing base and novel roles across tasks:

| setting | method | HDPE_LDPE | PP_PS |
| --- | --- | --- | --- |
| A | CE_shrinkage | 12.064 | 10.672 |
| A | EP_shrinkage | 8.847 | 8.581 |
| A | PCA64_shrinkage | 4.353 | 9.314 |
| B | CE_shrinkage | 5.947 | 6.469 |
| B | EP_shrinkage | 6.944 | 3.036 |
| B | PCA64_shrinkage | 0.956 | 4.611 |
| C | CE_shrinkage | 8.458 | 4.878 |
| C | EP_shrinkage | 7.136 | 2.222 |
| C | PCA64_shrinkage | 3.361 | 4.181 |
| D | CE_shrinkage | 11.006 | 10.986 |
| D | EP_shrinkage | 10.208 | 8.331 |
| D | PCA64_shrinkage | 4.169 | 10.158 |


| direction | head | CE_gap_pp | EP_gap_pp | EP_minus_CE_gap_pp |
| --- | --- | --- | --- | --- |
| C4_to_C8_gap | euclidean | -2.778 | -16.687 | -13.909 |
| C4_to_C8_gap | full | -8.990 | -11.236 | -2.245 |
| C4_to_C8_gap | shrinkage | -17.903 | -16.122 | 1.781 |
| C8_to_C4_gap | euclidean | -11.873 | -18.862 | -6.989 |
| C8_to_C4_gap | full | -7.933 | -10.969 | -3.035 |
| C8_to_C4_gap | shrinkage | -20.561 | -17.547 | 3.015 |


Primary: final 6-class session, mean of 5/10 shots and 200 matched episodes within each task/seed, then mean over 3 tasks x 3 seeds. All deltas are percentage points. Tasks share data; seeds and repeated episodes are not independent datasets or specimens. No significance claim is made. Group Macro-F1 averages global per-class F1, including cross-group false positives.

Frozen feature geometry is provided because objective-dependent feature scales change the effective strength of a fixed numeric ridge. Registration and prediction costs exclude preprocessing and encoding; no previous class state was recomputed. Mean per-model training and inference costs are reported below. Timing comparisons are descriptive measurements on the same CPU environment rather than a controlled hardware benchmark.

| objective | training_seconds | inference_median_ms_per_spectrum | optimizer_updates |
| --- | --- | --- | --- |
| CE | 77.616 | 0.279 | 860.000 |
| EP | 71.938 | 0.277 | 860.000 |


These findings delimit where episodic representation training and covariance shrinkage help under a reusable frozen-encoder protocol; they do not establish universal superiority over the spectrum/PCA reference. The defensible methodological contribution is the auditable separation of training-objective, metric-head, and reference-resolution effects in few-shot incremental registration. Conservative duplicate/record grouping does not establish specimen independence, both datasets were previously inspected, and C4/C8 are not independent-instrument validation. All results are exploratory. No further objectives, architectures, augmentations, or post-hoc hyperparameter searches were introduced following this final ablation.
