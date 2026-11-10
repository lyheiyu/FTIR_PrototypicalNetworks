# Descriptive feature geometry

| architecture | role | mean_norm | effective_rank_entropy_covariance | within_class_distance | between_class_distance | neighbor_consistency_excluding_self |
| --- | --- | --- | --- | --- | --- | --- |
| M0 | base_train | 15.028 | 2.014 | 4.426 | 11.730 | 0.986 |
| M0 | base_validation | 15.041 | 1.977 | 4.622 | 11.715 | 0.959 |
| M0 | novel_query_analysis_only | 14.695 | 2.234 | 3.932 | 5.258 | 0.975 |
| M1 | base_train | 131.012 | 4.462 | 46.736 | 129.633 | 0.999 |
| M1 | base_validation | 130.970 | 4.485 | 47.191 | 129.279 | 0.995 |
| M1 | novel_query_analysis_only | 134.411 | 4.623 | 50.799 | 94.728 | 0.997 |

Effective rank uses the entropy of the normalized covariance eigenvalues on centered features. Distances are Euclidean on unnormalized features. Neighbor consistency excludes self-neighbors. Novel-query geometry is retrospective analysis only, never used to fit transformations or select models.
