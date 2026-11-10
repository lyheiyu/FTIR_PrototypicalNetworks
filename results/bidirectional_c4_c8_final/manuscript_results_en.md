# Results

Six prespecified settings were evaluated using 2047 conservative paired record components, 3 class tasks and 3 seeds. C4- and C8-trained M1 encoders used matched 60-epoch cross-entropy budgets, with evaluation-query components excluded from training and checkpoint selection. Primary results average 5/10-shot conditions within each task/seed.

With source-domain base references, full-covariance M1 achieved 81.90% accuracy for C4-to-C8 and 82.99% for C8-to-C4 transfer; shrinkage achieved 77.25% and 75.28%. Thus, direction rankings depended on the classifier. Target-domain base references increased shrinkage accuracy to 94.43% and 94.68%, respectively, but required 20 labelled target base spectra per class. In the prior-free 1-NN control, mean absolute reference-bank and matched-target encoder-source effects were 11.60 and 0.99 percentage points, supporting a larger conditional reference-bank effect.

Shrinkage outperformed full covariance with within-domain or target-aligned base banks but not mixed-source banks. PCA64-shrinkage exceeded M1-shrinkage by 2.20–12.83 points across the four cross-source settings; the neural representation therefore showed no mean accuracy advantage over this fixed control. Novel classes were registered without encoder updates, although old-class recognition could decline through decision competition.

These results remain exploratory: record blocking does not verify specimen independence, datasets were previously inspected, and C4/C8 are not independent instruments. Repeated episodes were not independent specimens. Native-grid baseline processing and unequal relative regularisation of PCA64 versus M1-128 also limit causal interpretation.
