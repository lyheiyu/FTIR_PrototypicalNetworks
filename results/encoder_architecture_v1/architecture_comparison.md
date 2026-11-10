# Paired M0/M1 architecture comparison

C4 exploratory, base-only CE, identical fixed 1868-point legacy-cubic input, saved splits, explicit epoch orders, optimizer update budget and legal unseen episodes. 128-dimensional unnormalized ReLU embeddings; linear SVM C tuned only on base validation.

M0: compact three-convolution network with global average pooling. M1: 9/25/49-point branches, two factor-2 pools, 32-channel local convolutions, 8-channel projection and 32 contiguous position bins before the embedding. This tests the complete M1 design, not isolated causality of multiscale vs position.

## Unseen two-class accuracy (%)

| shot | M0_1nn | M0_svm | M1_1nn | M1_svm | raw_svm |
| --- | --- | --- | --- | --- | --- |
| 1 | 62.817 | 62.817 | 74.052 | 74.052 | 77.248 |
| 5 | 76.413 | 72.844 | 90.426 | 90.548 | 93.065 |
| 10 | 81.541 | 80.478 | 93.926 | 94.376 | 96.467 |
| 20 | 86.000 | 85.741 | 96.648 | 97.224 | 98.546 |

## Model and cost means

| architecture | parameters | training_seconds | optimizer_updates | inference_median_ms_per_spectrum | all_spectra_feature_extraction_seconds | model_file_bytes |
| --- | --- | --- | --- | --- | --- | --- |
| M0 | 47872.000 | 125.791 | 1340.000 | 0.309 | 0.891 | 221063.000 |
| M1 | 47400.000 | 135.461 | 1340.000 | 0.287 | 0.969 | 231860.000 |

Training timing includes fixed-epoch fitting, base validation and checkpoint writes. Inference is a warmed-up graph, batch 64, median over ten repeats, excluding preprocessing and classifier. Raw SVM head timings are in metrics.csv; shared preprocessing is recorded separately. No speed claim is made from parameter counts alone.
