# Conclusion

M1 meets the predeclared descriptive consistency gate. No additional architectures or losses were tried.

| head | mean_gain_pp | winning_trials | total_trials | positive_task_means | passes |
| --- | --- | --- | --- | --- | --- |
| svm | 15.801 | 9 | 9 | 3 | True |
| 1nn | 13.199 | 9 | 9 | 3 | True |

Gate: each of SVM and 1-NN must gain at least 1 percentage point on the 5/10-shot average, win at least 6 of 9 paired task-seed trials, and have positive means on all three tasks. This is a descriptive preregistered stop rule, not a significance test; task/seed samples overlap. C4 novel results have been inspected before; exact deduplication does not establish specimen independence. No independent confirmatory dataset has been evaluated.
