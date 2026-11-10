# 异构 FTIR 冻结迁移评估 V1

已完成 9 个既有 M1 checkpoint 的冻结评估；每种条件 200 次配对支持/查询抽样，1/5/10/20-shot。没有训练新 encoder、没有改 loss、没有使用 D4 query 选参数。

## 结论先读

**完整协方差消融后的关键结论：收缩在 D4 上不是最优距离头。** 5/10-shot 中，收缩相对原公式完整类协方差的总体 Accuracy，在 D4 固定九类任务差 -1.449 个百分点（改善 1/9），在跨域增量差 -5.587 个百分点（改善 1/6）。C4 上收缩更好、D4 上完整类协方差更好的反转，提示应研究协方差先验的可迁移性，而非直接推广原结论。详见文末完整消融和 full_class_mahalanobis_paired_summary.csv。

**跨域增量仍未解决。** 5-shot 最终阶段，即使采用总体更好的原公式完整类协方差，旧类 PP 召回率仍只有 0.067%，PS 为 96.233%。必须把逐类 base 保持作为后续硬性目标，不能仅报告新类或总体平均。

1. 既有 M1 可以通过明确、固定的部分光谱适配器处理 D4。107 个未测坐标用各 checkpoint 的合法 C4 base_train 坐标均值填补；这不等于恢复缺失峰，也不证明物理信号语义一致。
2. C4 的 5/10-shot 遮罩控制中，收缩方法 Accuracy 变化为 -3.205 个百分点（masked−original）。该控制只改变 query 的缺失坐标，支持库和已注册状态保持原样。
3. D4 固定九类适配中，收缩相对 M1 1-NN 的 5/10-shot 总体 Accuracy 平均差为 +3.780 个百分点，改善 9/9 组；C4→D4 增量中已加入新类 Accuracy 差为 +1.821 个百分点，改善 4/6 组，但总体 Accuracy 差为 -2.010 个百分点。收缩的新类收益没有转化为总体增量优势，不能把 C4 上的收益直接当作 D4 上的结论。
4. 固定 PP/PS 支持库的锚点中，D4 收缩 Accuracy 相对 masked C4 差 -42.250 个百分点。标签和分类头相同，但 D4 query 不是同一批实物；该差异包含域差异、信号语义及原生预处理区间差异，不能纯归因为缺失覆盖。

5. 同一 PP/PS 头的收缩方法在 C8 为 81.408%，在 D4 为 56.342%；欧氏头则分别为 77.817% 和 95.475%。数据集难度排名依赖分类头，不能概括为 D4 一定比 C8 更难。

## A. C4 内部结果与缺失覆盖敏感性

历史六类增量结果继续作为探索性参照。下面是相同 checkpoint、相同完整 C4 支持库下，原始 query 与 masked query 的配对变化：

| method | accuracy_gap_pp | prediction_agreement |
| --- | --- | --- |
| M1 1-NN | -0.368 | 0.982 |
| M1 Euclidean | 0.722 | 0.971 |
| M1 Shared | -0.710 | 0.976 |
| M1 Shrinkage | -3.205 | 0.956 |

prediction_agreement 是预测一致比例；变化单位为百分点。完整与遮罩都沿用原有输入尺度，不重新拟合 normalization、PCA 或协方差先验。历史完整 C4 数值见 c4_historical_reference.csv，逐 episode 遮罩结果见 coverage_mask_comparison.csv。

## B. C4→C8 跨分辨率迁移

| shot | M1 1-NN | M1 Euclidean | M1 Shared | M1 Shrinkage | Frozen PCA64 1-NN | Spectrum 1-NN | Spectrum SVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 66.669 | 71.273 | 72.102 | 62.713 | 73.680 | 73.944 | 78.527 |
| 5 | 72.340 | 75.560 | 76.070 | 68.516 | 77.359 | 77.853 | 83.848 |
| 10 | 72.993 | 76.782 | 76.736 | 73.823 | 77.666 | 78.272 | 85.951 |
| 20 | 73.538 | 77.801 | 77.054 | 78.037 | 78.020 | 78.611 | 87.438 |

表为最终六类 Accuracy（%）。5-shot 收缩方法的 Accuracy 为 68.516%，Macro-F1 为 67.314%。 支持样本来自原 C4；C4/C8 查询按同记录 ID、同标签配对，并按每类可匹配查询数的最小值取平衡子集。不能用全量 C4 分数直接减这个 C8 分数。正确的配对迁移差值（C8−对应 C4，跨全部 shot 的任务/种子均值）：

| method | mean | min | max |
| --- | --- | --- | --- |
| M1 1-NN | -20.179 | -30.084 | -14.741 |
| M1 Euclidean | -3.347 | -9.218 | 5.391 |
| M1 Shared | -14.052 | -20.829 | -4.267 |
| M1 Shrinkage | -24.054 | -30.245 | -20.189 |
| Frozen PCA64 1-NN | -15.831 | -19.371 | -11.013 |
| Spectrum 1-NN | -15.450 | -19.165 | -10.843 |
| Spectrum SVM | -7.982 | -10.697 | -5.091 |

C4/C8 属于同一数据家族的分辨率版本，不能称为独立仪器泛化。同编号配对只是记录级对应，尚无元数据证明两行来自同一实物或同一次采集，所以差值不能纯归因于分辨率。另存 c4_c8_record_alignment_audit.csv 检查同编号光谱的公共网格相似度；不按相似度挑选 query。base 查询来自曾参与 checkpoint 选择的验证集。若个别 episode 某类没有可匹配 query，两种分辨率同时跳过，匹配清单完整保存在每个 trial 的 c8_paired_queries.json。

## C. C4→D4 的 2+1+1 类增量

合法的六个 checkpoint：base 训练包含 PP、PS。头部仅保留这两类的原 C4 20-shot 支持；然后按 PE→PEVA 注册 D4 新类，每类五个固定池内抽取的独立 query。每次只更新新类均值/度量，已注册的 prototype 状态保持不变；1-NN/SVM 则保留全库并分别重建索引/重拟合 SVM。

最终四类总体 Accuracy（%）：

| shot | M1 1-NN | M1 Euclidean | M1 Shared | M1 Shrinkage | Frozen PCA64 1-NN | Spectrum 1-NN | Spectrum SVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 54.071 | 53.321 | 54.458 | 52.321 | 55.196 | 55.683 | 36.346 |
| 5 | 61.154 | 59.254 | 59.317 | 59.408 | 63.375 | 64.104 | 46.367 |
| 10 | 63.996 | 60.500 | 59.942 | 61.721 | 66.483 | 67.029 | 48.271 |
| 20 | 66.375 | 60.713 | 60.292 | 63.650 | 68.729 | 68.858 | 48.521 |

5-shot 收缩方法的 Accuracy 为 59.408%，Macro-F1 为 51.900%。

最终已加入新类 PE/PEVA Accuracy（%）：

| shot | M1 1-NN | M1 Euclidean | M1 Shared | M1 Shrinkage | Frozen PCA64 1-NN | Spectrum 1-NN | Spectrum SVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 65.058 | 65.342 | 63.800 | 67.217 | 67.042 | 67.600 | 67.967 |
| 5 | 80.067 | 77.442 | 74.550 | 82.283 | 84.117 | 85.150 | 88.267 |
| 10 | 86.083 | 79.750 | 75.792 | 87.508 | 90.608 | 91.417 | 92.233 |
| 20 | 91.192 | 80.217 | 76.467 | 92.933 | 95.192 | 95.417 | 92.983 |

base/current/previous/seen novel 的 Accuracy 和 Macro-F1 全部保存在 c4_to_d4_incremental_metrics.csv 和 summary_metrics.csv；base query 取 D4 的 PP/PS，不冒充 C4 内部 base 测试。

5-shot 各阶段的 base Accuracy（%）：

| session | M1 1-NN | M1 Euclidean | M1 Shared | M1 Shrinkage | Frozen PCA64 1-NN | Spectrum 1-NN | Spectrum SVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 76.367 | 95.475 | 70.908 | 56.342 | 55.700 | 55.367 | 22.800 |
| 1 | 42.492 | 41.358 | 45.817 | 39.433 | 42.950 | 43.533 | 4.617 |
| 2 | 42.242 | 41.067 | 44.083 | 36.533 | 42.633 | 43.058 | 4.467 |

新类高准确率不能替代 base 保持。这里的 base 下降发生在 encoder 和旧 prototype 参数完全不变时，反映跨域 query 与新增类竞争；不能称为训练覆盖了旧权重。三阶段平均总体指标另存 average_incremental_metrics.csv。

## D. D4 内部 few-shot 适配，encoder 仍来自 C4

| shot | M1 1-NN | M1 Euclidean | M1 Shared | M1 Shrinkage | Frozen PCA64 1-NN | Spectrum 1-NN | Spectrum SVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 47.375 | 47.375 | 45.204 | 45.204 | 53.179 | 53.763 | 53.763 |
| 5 | 64.325 | 56.136 | 55.063 | 67.609 | 70.951 | 71.748 | 75.237 |
| 10 | 70.748 | 57.399 | 57.106 | 75.025 | 77.502 | 78.296 | 81.700 |
| 20 | 76.002 | 58.023 | 58.427 | 80.633 | 82.494 | 83.496 | 85.885 |

5-shot 收缩方法的 Accuracy 为 67.609%，Macro-F1 为 66.658%。

表为固定九类任务的 Accuracy（%）。扩大类集的 11/10 类结果另存同一 d4_fewshot_metrics.csv 的 eligible 协议，不能沿变动类数画同一条主曲线。三种 encoder base 划分中，只有 holdout_pp_ps 的 D4 标签与 encoder 训练标签全部不重合；其三个种子的独立分表为 strict_label_disjoint_summary.csv。其余六个 checkpoint 已在 C4 训练见过 PP/PS 标签，但没有用 D4 数据训练。普通 PE 与 HDPE/LDPE 的化学关系也不能被“标签不重合”掩盖。

## 同标签、同支持库的 PP/PS 锚点

| scenario | M1 1-NN | M1 Euclidean | M1 Shared | M1 Shrinkage | Frozen PCA64 1-NN | Spectrum 1-NN | Spectrum SVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_C4 | 98.975 | 85.558 | 99.558 | 100.000 | 96.067 | 96.183 | 95.575 |
| anchor_C4_masked | 98.450 | 85.642 | 99.367 | 98.592 | 96.183 | 96.183 | 95.575 |
| anchor_C8 | 69.917 | 77.817 | 78.225 | 81.408 | 72.508 | 72.225 | 69.392 |
| anchor_D4 | 76.367 | 95.475 | 70.908 | 56.342 | 55.700 | 55.367 | 22.800 |

这里四种条件的分类头都只用原 C4 PP/PS 支持。C4 original/masked/C8 使用相同编号的 query 对；D4 是不同样本的相同文本标签。原始与 masked C4 直接估计缺失坐标敏感性，D4 与 masked C4 的剩余差异只能作为域差异诊断。该锚点没有新增 D4 支持，不能和 D4 内部适配分数混为一谈。

## 成本与冻结验证

cost_summary.csv 分开给出新类特征编码、局部注册、全库装配/重拟合 SVM 与缓存 query 推理。新 D4 支持的 M1 编码实际重放计时，均值/精度矩阵只注册新类。查询推理时间不含 encoder 与原始预处理；D4 内部 episode_head_fit_ms 是整个 episode 头部拟合成本，不能标成单类注册。
5-shot 两个新增阶段的平均成本（ms）：

| method | feature_encoding_ms | new_class_statistics_or_bank_ms | svm_full_refit_ms | total_update_ms | cached_query_prediction_ms |
| --- | --- | --- | --- | --- | --- |
| M1 1-NN | 3.148 | 0.013 |  | 3.475 | 0.228 |
| M1 Euclidean | 3.148 | 0.021 |  | 3.170 | 0.050 |
| M1 Shared | 3.148 | 0.016 |  | 3.164 | 0.186 |
| M1 Shrinkage | 3.148 | 0.316 |  | 3.464 | 0.120 |
| Frozen PCA64 1-NN | 3.335 | 0.006 |  | 3.592 | 0.219 |
| Spectrum 1-NN | 0.000 | 0.064 |  | 0.450 | 0.605 |
| Spectrum SVM | 0.000 | 0.068 | 0.930 | 0.998 | 0.374 |

所有 checkpoint 的文件 SHA 与权重 SHA 前后完全相同，均为 47,400 参数，新增训练更新数为零。PCA64、base prior 和旧实验脚本均通过 hash 复核。

## 下一步算法研究判断

完整类协方差控制显示，优先研究“什么时候应使用 C4 base 协方差先验”及跨域 base/novel 分数可比性更有证据基础。当前固定收缩公式没有随目标域变化，可能产生先验负迁移；这是一项待独立验证的机制假设，尚非因果结论。

遮罩 C4 存在可见敏感性，覆盖感知模型值得作为后续独立实验假设；但现有结果不能证明它能修复 D4 的所有域差异。
收缩在 D4 的收益存在条件差异，应优先检验 C4 协方差先验的可迁移性、跨域 base 类校准和类内/类间结构，不能直接照搬 C4 的结论。尤其新类改善而总体下降时，下一步应把 base 保持作为硬性评估目标。
prototype-oriented encoder 训练只能作为未来受控比较的候选方向；本轮没有训练它，也没有证据证明某个新增 loss 会解决当前问题。若原谱/PCA 明显优于 embedding，这提示当前特征迁移不足；若两者都弱，则信号语义与采样条件更需优先核实。
当前不能把缺失覆盖、未知物理信号和类别判别难度的影响完全因果分离。需要有已知信号语义、已知实物分组、未被查看的独立数据，才能开展确认性评价。

## 文件与图

- figures/transfer_overview.png/pdf：分辨率、D4 适配、增量新类与遮罩影响。
- figures/PP_PS_transfer_anchor.png/pdf：固定两类头的跨数据集锚点。
- figures/*_5shot_CM.png/pdf：C8 六类、D4 九类与增量四类的逐类别混淆矩阵及原计数 CSV。
- per_class_results.csv：按 task/seed 分开的累计混淆矩阵召回率；重复查询计数不代表独立样本数量。
- trial_metrics.csv、d4_paired_method_deltas_pp.csv：先对重复抽样求均值的配对结果，不把 200 个 episode 当作 200 个独立训练模型。
- protocol_audit.json、frozen_encoder_verification.csv：输入、划分、来源和冻结核查。

所有数字属于探索性评估。D4 部分光谱结果基于固定均值填补；记录 ID 匹配不等于实物独立；原生轴不同使零 exact hash 匹配不能排除重采样/重复实物。未覆盖测量峰、未知信号单位和 morphotype 标签身份仍是未解决的数据限制。

## 附：保留完整类协方差 Mahalanobis 消融

沿用上一轮完整类协方差实现，在完全相同的外部 support/query、冻结特征和 C4 支持库上补测。两组固定 ridge 为 0.001 与原代码有效值 0.002，均不使用 base 协方差 prior；收缩则额外使用 C4 base_train prior。没有按 D4 query 调参，没有重新推理或训练 encoder。
协方差为类内残差外积除以 n，完整 128×128 非对角矩阵。少于 129 个支持时无正则经验协方差均奇异；这里比较的是带明确 ridge 的完整类协方差。增量仅注册新类；首个 repeat 的所有 shot/阶段逐元素验证旧均值和精度矩阵未变。

严重的逐类限制：5-shot 最终增量阶段，原公式完整类协方差对 D4 base PP 的召回率为 0.067%，对 PS 为 96.233%。PP 几乎全部被新增 PE/PEVA 吸引。完整类协方差总体优于收缩，不等于已实现可靠的跨域旧类保持。

### D4_within: 总体 Accuracy（%）

| shot | M1 1-NN | M1 Class full .002 | M1 Class full .001 | M1 Shrinkage | Spectrum SVM |
| --- | --- | --- | --- | --- | --- |
| 1 | 47.375 | 47.375 | 47.375 | 45.204 | 53.763 |
| 5 | 64.325 | 68.860 | 68.860 | 67.609 | 75.237 |
| 10 | 70.748 | 76.672 | 76.668 | 75.025 | 81.700 |
| 20 | 76.002 | 81.981 | 81.967 | 80.633 | 85.885 |

### C4_to_D4_incremental: 总体 Accuracy（%）

| shot | M1 1-NN | M1 Class full .002 | M1 Class full .001 | M1 Shrinkage | Spectrum SVM |
| --- | --- | --- | --- | --- | --- |
| 1 | 54.071 | 51.796 | 51.642 | 52.321 | 36.346 |
| 5 | 61.154 | 65.038 | 65.038 | 59.408 | 46.367 |
| 10 | 63.996 | 67.267 | 67.275 | 61.721 | 48.271 |
| 20 | 66.375 | 67.896 | 67.979 | 63.650 | 48.521 |

### C8_resolution: 总体 Accuracy（%）

| shot | M1 1-NN | M1 Class full .002 | M1 Class full .001 | M1 Shrinkage | Spectrum SVM |
| --- | --- | --- | --- | --- | --- |
| 1 | 66.669 | 58.694 | 58.642 | 62.713 | 78.527 |
| 5 | 72.340 | 66.092 | 65.995 | 68.516 | 83.848 |
| 10 | 72.993 | 72.770 | 72.681 | 73.823 | 85.951 |
| 20 | 73.538 | 77.723 | 77.672 | 78.037 | 87.438 |

5/10-shot 收缩减完整类协方差，单位为百分点。先对每个任务/种子重复抽样求均值，再配对汇总：

| scenario | metric | reference | mean | min | max | count | positive_trials |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C4_to_D4_incremental | all_accuracy | M1_full_class_legacy002 | -5.587 | -20.575 | 1.050 | 6 | 1 |
| C4_to_D4_incremental | all_accuracy | M1_full_class_ridge001 | -5.592 | -20.575 | 1.050 | 6 | 1 |
| C4_to_D4_incremental | all_macro_f1 | M1_full_class_legacy002 | -5.348 | -21.739 | 0.976 | 6 | 1 |
| C4_to_D4_incremental | all_macro_f1 | M1_full_class_ridge001 | -5.349 | -21.740 | 0.976 | 6 | 1 |
| C4_to_D4_incremental | base_accuracy | M1_full_class_legacy002 | -10.946 | -41.150 | -0.300 | 6 | 0 |
| C4_to_D4_incremental | base_accuracy | M1_full_class_ridge001 | -10.963 | -41.175 | -0.300 | 6 | 0 |
| C4_to_D4_incremental | seen_novel_accuracy | M1_full_class_legacy002 | -0.229 | -3.675 | 2.400 | 6 | 3 |
| C4_to_D4_incremental | seen_novel_accuracy | M1_full_class_ridge001 | -0.221 | -3.675 | 2.400 | 6 | 4 |
| C8_resolution | all_accuracy | M1_full_class_legacy002 | 1.739 | -4.450 | 6.508 | 9 | 6 |
| C8_resolution | all_accuracy | M1_full_class_ridge001 | 1.831 | -4.095 | 6.201 | 9 | 6 |
| C8_resolution | all_macro_f1 | M1_full_class_legacy002 | 2.876 | -2.312 | 7.486 | 9 | 8 |
| C8_resolution | all_macro_f1 | M1_full_class_ridge001 | 2.956 | -1.929 | 7.509 | 9 | 7 |
| C8_resolution | base_accuracy | M1_full_class_legacy002 | -14.636 | -26.955 | -4.602 | 9 | 0 |
| C8_resolution | base_accuracy | M1_full_class_ridge001 | -14.589 | -26.651 | -5.063 | 9 | 0 |
| C8_resolution | seen_novel_accuracy | M1_full_class_legacy002 | 34.488 | 24.376 | 41.652 | 9 | 9 |
| C8_resolution | seen_novel_accuracy | M1_full_class_ridge001 | 34.671 | 24.409 | 41.823 | 9 | 9 |
| D4_within | all_accuracy | M1_full_class_legacy002 | -1.449 | -2.872 | 0.078 | 9 | 1 |
| D4_within | all_accuracy | M1_full_class_ridge001 | -1.448 | -2.872 | 0.083 | 9 | 1 |
| D4_within | all_macro_f1 | M1_full_class_legacy002 | -1.498 | -3.353 | -0.091 | 9 | 0 |
| D4_within | all_macro_f1 | M1_full_class_ridge001 | -1.496 | -3.353 | -0.085 | 9 | 0 |
| D4_within | base_accuracy | M1_full_class_legacy002 | -0.917 | -4.125 | 0.400 | 6 | 1 |
| D4_within | base_accuracy | M1_full_class_ridge001 | -0.912 | -4.100 | 0.400 | 6 | 1 |
| D4_within | seen_novel_accuracy | M1_full_class_legacy002 | -1.502 | -3.021 | 1.279 | 9 | 1 |
| D4_within | seen_novel_accuracy | M1_full_class_ridge001 | -1.500 | -3.014 | 1.279 | 9 | 1 |

完整对照保存在 full_class_mahalanobis_control_metrics.csv，分类别和混淆矩阵位于 full_mahalanobis_control/，综合比较图为 figures/full_class_mahalanobis_transfer_CM.png/pdf。原七种方法结果未被覆盖。
