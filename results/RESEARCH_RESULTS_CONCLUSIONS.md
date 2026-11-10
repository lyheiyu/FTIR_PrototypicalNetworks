# 新结果结论：冻结特征诊断与 C4 类别增量

分析对象：frozen_diagnosis_v1、c4_incremental_v1，均有 COMPLETE 标记。
前者为 2 数据集×3 seed×4 shot×200 episode×7 方法，共 33600 条指标；后者为 3 类别任务×3 seed×4 shot×20 repeat×3 session×8 方法，共 17280 条指标。

## 1. 冻结特征诊断：当前表征是优先检查项

Accuracy（%，各 seed/episode 均值）：

| 数据 | Shot | 预处理谱 SVM | PCA64 SVM | Encoder SVM | Encoder 每类 Mahalanobis |
| --- | ---: | ---: | ---: | ---: | ---: |
| C4 | 5 | 87.64 | 87.62 | 77.80 | 77.71 |
| C4 | 20 | 96.38 | 96.35 | 89.56 | 89.69 |
| C8 | 5 | 91.12 | 91.13 | 78.91 | 79.65 |
| C8 | 20 | 98.39 | 98.40 | 90.61 | 91.64 |

- PCA64 几乎保留当前协议下 SVM 的判别性能。不能简单归因于“所有降维都会丢失太多信息”。
- Encoder SVM 与每类 Mahalanobis 接近，换成 SVM 头并未恢复原谱性能，说明不能只优化 Mahalanobis；当前表示、训练方式和特征尺度应优先检查。
- 这是强诊断线索，不是证明所有可能分类器均无法从 embedding 恢复信息；SVM 类型、搜索范围和归一化等仍有边界。
- 本轮 raw_svm 已通过 validation 调 C，而旧 fewshot_main 固定 C=1；对应分数变化不能解读为新 encoder 改善。
- 此处“原谱”实际是 legacy-preprocessed spectrum：反转、airPLS、归一化、2000 点 cubic，不能称未经处理 native-grid SVM。

## 2. C4 类别增量：收缩有效，但未胜过简单模板方法

最终 Session 2 六类 Accuracy（%，3 任务×3 seed×20 repeat 的等权均值）：

| Shot | 光谱 1-NN | 光谱 SVM | Encoder 共享 prior | Encoder 独立类协方差 | Encoder 收缩协方差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 84.38 | 76.31 | 75.71 | 62.71 | 75.66 |
| 5 | 91.30 | 92.20 | 80.29 | 78.80 | 83.35 |
| 10 | 93.57 | 94.80 | 81.64 | 81.85 | 84.76 |
| 20 | 95.14 | 96.08 | 82.52 | 85.49 | 87.30 |

- 收缩方案在 5/10/20-shot 最终 Session 分别较独立类协方差提高 4.55/2.91/1.81 个百分点，且三个类别任务的 5-shot 最终均值均改善，值得保留作为候选增量组件。
- 收缩不是所有条件最优：1-shot 与共享 prior 接近，略低于共享方案；这不构成统计显著结论。
- 光谱 1-NN 同样无需优化参数即可添加类别，且当前显著强于 encoder 增量头。“不重训 Encoder”不足以单独论证 learned encoder 的价值。
- 5-shot AIA 为：收缩 89.05%、独立类协方差 85.14%、共享 prior 87.70%、光谱 1-NN 94.18%、光谱 SVM 95.06%。AIA 高于最终 Session 是因为较早 session 类别更少、更容易；两者不可混用。

## 3. 1-shot 的旧新类不平衡尤其明显

独立类协方差方法在 Session 1 的 base Accuracy 为 91.85%，当前新类 Accuracy 却只有 5.11%；收缩方法的当前新类 Accuracy 为 66.33%。

不能据“旧类不掉分”认定增量成功。该条件下新类 singleton 的类内协方差为零，距离只依赖 ridge；base 类有 20 个 exemplar，拥有非零类协方差，两边距离尺度不同，容易偏向旧类。数据支持存在严重偏置，具体成因仍需按类距离、范数和协方差谱诊断。固定共享 prior/收缩可以缓解，但不能保证彻底解决。

metrics.csv 中 new_accuracy 指当次最新加入的类别，不是所有 novel 类合并的精度。Session 2 的 old 集合已经包括 Session 1 的 novel 类。应结合逐类 recall 和 CM 阅读。

## 4. 任务差异不可被总体均值掩盖

5-shot 收缩方法的最终 Accuracy：

- 留出 HDPE/LDPE：75.31%；光谱 1-NN 为 91.22%。
- 留出 PP/PS：85.63%；光谱 1-NN 为 89.54%。
- 留出 PET/PVC：89.09%；光谱 1-NN 为 93.13%。

当前 encoder 对未见 HDPE/LDPE 的迁移尤其需要检查，不能以一个较容易的留出任务代表整体泛化。

## 5. 证据边界

- 九个 incremental trial 的 manifest 和训练记录显示 encoder train 只含指定 base 类；冻结文件 hash 均保持不变。
- 本轮 groups=null：只完成了文件内精确去重，没有完成真实 specimen 分组，不能称严格样品独立泛化。
- 每类固定 query 只有 15 条，20 个 repeat 复用这些 query；它们并非 20 份独立测试集。
- 三个任务和三个 seed 的样本/类别有重叠，不能当作九份完全独立数据；当前差距先按描述性发现报告。
- 没有 v2 的多 harmonizer/网格实验，不足以认定 cubic 是瓶颈；也没有实际局部 Feature Bank 更新成本结果，不足以认定比 SVM/1-NN 更新更快。

## 6. 与 v2 计划对应的下一步

1. 保留当前结果为 V1，完成信号语义和样品身份审计。
2. 先做同一开发协议下 Linear/Cubic/PCHIP×少量物理网格的 SVM 和保真比较，不将新输入直接送旧 encoder 判优。
3. 先以 PCA64 与光谱 1-NN 作为低成本强基线；在固定输入后，用控制预算的表征实验检验位置保留结构、L2/cosine、episodic 或 SupCon。一次只引入可归因的变化。
4. 保留收缩头，补低 shot 旧新类偏置分析及持久 Feature Bank；不要再把主要精力放在仅调整独立类协方差的 ridge 上。
5. 新 pipeline 与调参只用合法开发数据；当前已被分析的测试结果属于探索性证据，后续确认性评估应预先锁定协议和选择规则。
