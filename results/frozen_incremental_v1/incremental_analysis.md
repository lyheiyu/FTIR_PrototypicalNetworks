# Frozen C4 4+1+1 evaluation

Reused architecture-V1 M0/M1 checkpoints, splits, preprocessing and exact novel support/query episodes. Encoder training calls: zero. Five existing heads: raw SVM, M0/M1 linear SVM and Euclidean 1-NN. Each head receives the same fixed 20 base-training exemplars/class and K support per arrived novel class. SVM C is reused from archived base-only tuning, with no query retuning. The full SVM refit uses all retained seen exemplars (80, 80+K, 80+2K rows), not all historical base-training data.

**Base query limitation:** base queries are sampled from archived base validation. Those rows already participated in checkpoint/C selection. Base and all-seen metrics therefore are validation-reuse exploratory estimates, not independent held-out-test performance. Novel queries remain in their original disjoint pools.

Accuracy for a group conditions on its true labels. Group Macro-F1 averages its global one-vs-rest class F1 over the entire seen-class confusion matrix, including false positives from other groups. Previously added novel excludes base classes and the current class; it is empty until session 2. Current novel is undefined in session 0. All-seen Macro-F1 is standard multiclass Macro-F1.

## Final session 1-shot (%)

| method | base_accuracy | previous_novel_accuracy | current_novel_accuracy | seen_novel_accuracy | all_accuracy | all_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| M0_1nn | 90.422 | 43.937 | 26.141 | 35.039 | 71.961 | 68.053 |
| M0_svm | 90.298 | 18.515 | 1.411 | 9.963 | 63.520 | 54.946 |
| M1_1nn | 98.091 | 63.537 | 43.863 | 53.700 | 83.294 | 80.541 |
| M1_svm | 98.421 | 66.570 | 44.522 | 55.546 | 84.130 | 81.524 |
| raw_svm | 87.906 | 64.456 | 27.885 | 46.170 | 73.994 | 70.363 |

## Final session 5-shot (%)

| method | base_accuracy | previous_novel_accuracy | current_novel_accuracy | seen_novel_accuracy | all_accuracy | all_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| M0_1nn | 88.858 | 63.896 | 58.078 | 60.987 | 79.568 | 78.517 |
| M0_svm | 90.201 | 59.174 | 43.900 | 51.537 | 77.313 | 74.738 |
| M1_1nn | 97.550 | 82.926 | 80.930 | 81.928 | 92.343 | 92.021 |
| M1_svm | 98.039 | 86.381 | 76.707 | 81.544 | 92.541 | 92.182 |
| raw_svm | 96.292 | 90.111 | 82.004 | 86.057 | 92.880 | 92.664 |

## Final session 10-shot (%)

| method | base_accuracy | previous_novel_accuracy | current_novel_accuracy | seen_novel_accuracy | all_accuracy | all_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| M0_1nn | 87.894 | 71.319 | 69.967 | 70.643 | 82.144 | 81.675 |
| M0_svm | 89.134 | 72.700 | 61.122 | 66.911 | 81.727 | 80.836 |
| M1_1nn | 97.197 | 89.230 | 90.178 | 89.704 | 94.699 | 94.628 |
| M1_svm | 97.770 | 91.004 | 87.770 | 89.387 | 94.976 | 94.885 |
| raw_svm | 95.975 | 93.874 | 90.959 | 92.417 | 94.789 | 94.716 |

## Final session 20-shot (%)

| method | base_accuracy | previous_novel_accuracy | current_novel_accuracy | seen_novel_accuracy | all_accuracy | all_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| M0_1nn | 86.509 | 78.078 | 80.478 | 79.278 | 84.099 | 83.919 |
| M0_svm | 87.694 | 81.033 | 74.863 | 77.948 | 84.446 | 84.158 |
| M1_1nn | 96.852 | 93.733 | 95.630 | 94.681 | 96.128 | 96.122 |
| M1_svm | 97.480 | 95.296 | 94.893 | 95.094 | 96.685 | 96.672 |
| raw_svm | 96.291 | 97.004 | 96.304 | 96.654 | 96.412 | 96.385 |

Full group accuracy/Macro-F1 for every session: `summary_metrics.csv`. Average incremental accuracy is reported both across sessions 0/1/2 and across 1/2 only. Task/seed means are equally weighted; repeated episodes are not independent training runs.
