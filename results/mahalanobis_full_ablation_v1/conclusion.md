# 完整类协方差 Mahalanobis 补充对照 V1

已完成 C4 三任务 × 三种子 × 四种 shot × 200 个配对 episode，包含全部三个增量阶段。M1 encoder 保持原来的 47,400 参数并完全冻结，本次训练步骤为零。

本轮结论：在预定的 5/10-shot 主要比较中，收缩相对原公式完整类协方差的总体 Accuracy 提高 4.958 个百分点，新类 Accuracy 提高 15.164 个百分点，九组配对总体 Accuracy 改善 9/9。完整类协方差随 shot 增加明显改善，但其 5/10-shot 平均总体 Accuracy 相对 1-NN 仍差 2.360 个百分点。因此支持进一步研究收缩方法，尚不能声称原始完整类协方差在少样本条件下全面优于强基线。

## 先区分两个问题

原来的 shared 和 shrinkage 都使用完整 128×128 矩阵。本次补充的是每类独立估计、没有向 base prior 收缩的完整协方差对照。

你的原始方法使用每类均值和每类协方差。它是否优于欧氏原型/共享协方差，与后加的收缩估计是否进一步改善它，是两个独立问题。之前缺少这一项，不能充分归因于收缩改进；也不能因此否定原始方法。

## 最终六类 Accuracy（%）

| shot | 1-NN | Class full (legacy .002) | Class full (ridge .001) | Euclidean | Shared full | Class shrinkage | SVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 83.294 | 74.292 | 74.234 | 73.447 | 85.617 | 87.688 | 84.13 |
| 5 | 92.343 | 88.17 | 88.104 | 76.651 | 89.831 | 94.998 | 92.541 |
| 10 | 94.699 | 94.151 | 94.127 | 77.549 | 90.822 | 97.24 | 94.976 |
| 20 | 96.128 | 96.641 | 96.628 | 78.168 | 91.252 | 98.953 | 96.685 |

## 最终六类 Macro-F1（%）

| shot | 1-NN | Class full (legacy .002) | Class full (ridge .001) | Euclidean | Shared full | Class shrinkage | SVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80.541 | 67.946 | 67.862 | 71.696 | 82.78 | 86.609 | 81.524 |
| 5 | 92.021 | 87.054 | 86.979 | 75.495 | 88.37 | 94.954 | 92.182 |
| 10 | 94.628 | 93.946 | 93.92 | 76.575 | 89.587 | 97.22 | 94.885 |
| 20 | 96.122 | 96.621 | 96.607 | 77.301 | 90.121 | 98.947 | 96.672 |

## 主要配对比较：5/10-shot

先对每个任务/种子内的 episode 与 5/10-shot 求均值，再对九组配对值汇总。下面的差值单位为百分点。正值表示前者更好。

| a | b | all_accuracy | all_macro_f1 | seen_novel_accuracy | positive_all_accuracy_trials |
| --- | --- | --- | --- | --- | --- |
| M1_full_class_legacy002 | M1_1nn | -2.3605 | -2.8244 | -8.5796 | 0 |
| M1_full_class_legacy002 | M1_proto_shared | 0.8343 | 1.5212 | 5.8407 | 7 |
| M1_full_class_ridge001 | M1_proto_euclidean | 14.016 | 14.4148 | 4.6343 | 9 |
| M1_full_class_ridge001 | M1_proto_shared | 0.7895 | 1.4707 | 5.7046 | 7 |
| M1_proto_shrinkage | M1_full_class_legacy002 | 4.9583 | 5.5871 | 15.1639 | 9 |
| M1_proto_shrinkage | M1_full_class_ridge001 | 5.0031 | 5.6376 | 15.3 | 9 |

## 如何解释

- Class shrinkage 对比 Class full (ridge .001)：总体 Accuracy +5.003，Macro-F1 +5.638，已加入新类 Accuracy +15.300；总体 Accuracy 改善 9/9 组。
- Class shrinkage 对比 Class full (legacy .002)：总体 Accuracy +4.958，Macro-F1 +5.587，已加入新类 Accuracy +15.164；总体 Accuracy 改善 9/9 组。
- Class full (ridge .001) 对比 Shared full：总体 Accuracy +0.790，Macro-F1 +1.471，已加入新类 Accuracy +5.705；总体 Accuracy 改善 7/9 组。
- Class full (legacy .002) 对比 Shared full：总体 Accuracy +0.834，Macro-F1 +1.521，已加入新类 Accuracy +5.841；总体 Accuracy 改善 7/9 组。
- Class full (ridge .001) 对比 Euclidean：总体 Accuracy +14.016，Macro-F1 +14.415，已加入新类 Accuracy +4.634；总体 Accuracy 改善 9/9 组。
- Class full (legacy .002) 对比 1-NN：总体 Accuracy -2.360，Macro-F1 -2.824，已加入新类 Accuracy -8.580；总体 Accuracy 改善 0/9 组。

不能把某个 shot 的均值优势解释为所有任务、种子、shot 都获胜；逐组结果保存在 paired_primary_5_10shot_pp.csv。

## 公式和可比性

原代码估计协方差时加 0.001I，计算距离时又加 0.001I，因此原公式实际是 C+0.002I。本次同时保留 C+0.001I 的同正则对照。复现的是原始距离公式，使用当前冻结 M1 的 float64 缓存特征；不是旧 float32 Torch 训练流程的逐位复现。省略平方根不改变最近类判决。

128 维下协方差秩最多为 n−1，所有 1/5/10/20-shot 类协方差均不可直接求逆。完整类协方差这里有明确 ridge，未用伪逆冒充无正则逆。

所有方法使用相同 support/query。完整类协方差只使用该类 support；共享/收缩还使用已存档 base_train 的类内协方差 prior。因此结果体现 prior 与收缩策略的联合价值，不能声称额外 base 数据访问已被完全控制。

## 注册、推理与存储

| method | head_update_ms | cached_query_prediction_ms | total_numeric_state_bytes |
| --- | --- | --- | --- |
| M1_full_class_legacy002 | 0.2667 | 0.3806 | 726572.0 |
| M1_full_class_ridge001 | 0.2805 | 0.3864 | 726572.0 |
| M1_proto_shared | 0.0188 | 0.4337 | 136748.0 |
| M1_proto_shrinkage | 0.3021 | 0.3781 | 857644.0 |

5-shot 的成本表平均两个新增阶段。注册只计算新类；查询推理使用缓存特征。数字不包含 encoder 推理、预处理、Python 开销，也不是端到端延迟。不同阶段类别数不同，表内存是两个阶段均值。最终六类完整类协方差头为 792,624 bytes；收缩头为 923,696 bytes，多存一份 base prior。两者仍需六个完整逆协方差矩阵。

本次完整类协方差头与收缩头都在注册时缓存逆矩阵；原 Torch 脚本是在查询/类别循环中反复求逆。因此这里测的是高效实现下各距离公式的成本，不能当作原始脚本的运行耗时。

## 图与实验边界

- figures/full_mahalanobis_comparison.png/pdf：各 shot 总体与新类 Accuracy。
- figures/*_final_CM.png/pdf：三个任务各自的 5/10-shot 混淆矩阵，六类分别展示；同时保存计数 CSV。
- final_per_class_recall.csv：按 shot、方法、类别的最终召回率。
- summary.csv 与 trial_metrics.csv：各阶段全部分组指标。

这是已反复查看过的 C4 探索性结果。base query 复用先前验证集，实物重复测量分组未知。不能据此声称独立数据泛化、确认性显著性或所有 FTIR 数据上的优越性。原 checkpoint、历史结果和预处理均未修改。
