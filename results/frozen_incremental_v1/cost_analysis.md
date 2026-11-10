# Cost conditions

| shot | method | local_registration_ms | memory_assembly_ms | svm_full_refit_ms | knn_index_build_ms | total_head_update_ms | predict_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | M0_1nn | 2.037 | 0.066 | nan | 0.393 | 2.496 | 2.136 |
| 1 | M0_svm | 2.037 | 0.165 | 1.003 | nan | 3.205 | 0.317 |
| 1 | M1_1nn | 1.844 | 0.066 | nan | 0.391 | 2.301 | 2.053 |
| 1 | M1_svm | 1.844 | 0.066 | 0.955 | nan | 2.866 | 0.293 |
| 1 | raw_svm | 0.007 | 0.473 | 3.024 | nan | 3.504 | 2.109 |
| 5 | M0_1nn | 5.017 | 0.060 | nan | 0.370 | 5.447 | 2.399 |
| 5 | M0_svm | 5.017 | 0.148 | 1.002 | nan | 6.168 | 0.293 |
| 5 | M1_1nn | 4.506 | 0.061 | nan | 0.367 | 4.934 | 2.321 |
| 5 | M1_svm | 4.506 | 0.062 | 0.941 | nan | 5.509 | 0.287 |
| 5 | raw_svm | 0.013 | 0.449 | 2.711 | nan | 3.173 | 1.667 |
| 10 | M0_1nn | 7.590 | 0.051 | nan | 0.313 | 7.954 | 2.131 |
| 10 | M0_svm | 7.590 | 0.136 | 0.983 | nan | 8.708 | 0.253 |
| 10 | M1_1nn | 7.048 | 0.054 | nan | 0.320 | 7.422 | 2.031 |
| 10 | M1_svm | 7.048 | 0.054 | 0.846 | nan | 7.948 | 0.257 |
| 10 | raw_svm | 0.019 | 0.446 | 2.625 | nan | 3.089 | 1.546 |
| 20 | M0_1nn | 10.824 | 0.054 | nan | 0.315 | 11.193 | 2.110 |
| 20 | M0_svm | 10.824 | 0.152 | 1.171 | nan | 12.147 | 0.273 |
| 20 | M1_1nn | 10.622 | 0.056 | nan | 0.312 | 10.990 | 2.007 |
| 20 | M1_svm | 10.622 | 0.057 | 0.906 | nan | 11.585 | 0.266 |
| 20 | raw_svm | 0.031 | 0.493 | 2.931 | nan | 3.454 | 1.616 |

Costs are milliseconds. Local registration is the warm frozen encoding of K arriving spectra plus copying that class into an append-only bank. Scoring uses archived features; timed re-encoding is a replay of the same support under the frozen checkpoint, checked against saved features. Raw spectra require no encoder. All costs exclude shared row-local preprocessing. Bank materialization and classifier fitting are separate and required for serving; local append alone is not the complete update. SVM is fully rebuilt on all retained seen exemplars, including previous novel support. 1-NN uses the existing sklearn head, whose index preparation is measured separately. total_head_update_ms includes encoding, append, materialization and refit/index build. Prediction uses 75 queries in session 1 and 90 in session 2, excludes cached query feature extraction, and should not be presented as end-to-end inference. Timing is sequential on one CPU with four TensorFlow threads; it is not a production throughput benchmark.
