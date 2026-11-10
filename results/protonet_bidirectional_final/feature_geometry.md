# Feature geometry

Source base-training embeddings only; no query-based fitting. Same numeric ridge does not equal same relative regularization across CE/EP or PCA64/128-dimensional M1.

| objective | source | mean_training_norm | mean_residual_variance | ridge_over_mean_variance | effective_rank |
| --- | --- | --- | --- | --- | --- |
| CE | c4 | 110.786 | 12.180 | 1.216e-04 | 6.494 |
| CE | c8 | 105.580 | 18.016 | 2.412e-04 | 4.910 |
| EP | c4 | 10.521 | 0.029 | 0.036 | 15.153 |
| EP | c8 | 8.422 | 0.024 | 0.045 | 12.628 |
