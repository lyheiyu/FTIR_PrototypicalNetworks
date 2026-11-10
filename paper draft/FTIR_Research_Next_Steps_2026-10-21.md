# 研究计划对照与近期执行顺序

对照文件：FTIR_Reusable_FewShot_Research_Plan.md。原计划不修改；本文明确已有证据、协议缺口和下一步任务。

## 一、目前的主要差距是任务定义，不只是 Accuracy

已完成：C4/C8 独立去重；共同波数轴；分层 train/validation/test；监督训练 encoder；冻结 embedding；配对 few-shot 距离比较、原始谱 SVM/KNN、逐类指标和 CM。

未完成：encoder 未见过的类别；连续 session 中旧新类共同分类；跨分辨率冻结迁移；独立设备/批次的泛化；更新时间、推理时间、存储和需要保留的旧数据量。

现有 encoder 在训练时见过全部六类。因此当前约 90% 的 20-shot 结果是已知类别的少样本适配，不是类别增量新类结果。旧 encoder 可以继续用于诊断，不能作为新类未见的 encoder 使用。

## 二、计划中应补充或纠正的地方

1. C4/C8 实际都是 PET/HDPE/PVC/LDPE/PP/PS 六类，不是不同类别子集；应称同一来源的不同分辨率版本。只有核实采集设备、批次和实物对应关系后，才能进一步声明跨设备/批次或跨实物独立性。
2. 第一阶段不要同时投入自监督、监督对比、episodic 和重建联合损失。先明确当前表示是否有问题，再一次只改变一个可解释因素。联合目标暂保留为后续候选。
3. 诊断 Encoder+SVM 只能说明当前已知类别中的判别信息，不证明它能泛化到未见类别。诊断与正式新类别实验必须分开解释。
4. Raw 1-NN、Raw Prototype、PCA+Prototype 同样可以通过添加模板而不重训 encoder。要证明 learned encoder 的必要性，应加入这些低成本增量基线，不能只比较需要重新拟合的 SVM。
5. 冻结网络仍可能因为新类加入竞争而降低旧类准确率；这是需要实测的类别干扰，不能根据参数不变断言没有性能遗忘。
6. 先用真实 K-shot support，不加入 EMSA，保持协议清晰。后续增强不得利用新类别的完整训练池、query 标签或未来 session 的统计量。

## 三、立即执行：不训练新网络的冻结特征诊断

已确认六个 dataset/seed 目录均有 encoder.h5、embeddings.npz、split_manifest.csv 和保存的 1/5/10/20-shot episode 索引；根目录有 C4/C8 的预处理缓存。无需重新提取 embedding。

新增比较：

| 输入 | 分类器 | 意义 |
| --- | --- | --- |
| 原始预处理谱 | SVM、1-NN、Prototype | 原始信息与无需优化的模板基线 |
| PCA 特征 | SVM、Prototype | 低维线性表示对照 |
| 冻结 embedding | SVM、1-NN、Prototype、现有 Mahalanobis | 区分表示与分类规则 |

必须复用已有 test episodes，原始谱、PCA 和 embedding 按同一 manifest 索引对齐。PCA 只在 encoder 的 train split 拟合；若使用足够的训练集拟合 PCA，应明确它与仅 support 拟合 PCA 是两种标注/无标注信息预算。SVM C、PCA 维数、可选标准化和 L2 特征变换只通过 validation 确定。为避免继续对 test 优化，把本轮诊断标注为探索性，后续正式增量协议预先固定。

看三个差距，而不只看一个冠军：

- embedding SVM 对 raw/PCA SVM：当前表征的判别信息。
- embedding SVM 对 embedding Prototype/Mahalanobis：分类规则的影响。
- raw/PCA 模板方法对 learned embedding 模板方法：encoder 是否带来真实收益。

若 embedding SVM 很弱，才优先检查 encoder。原始峰位置是否被多次 pooling/global average pooling 淡化是一个待验证假设；不能仅凭架构直接断言。先做位置保留的监督 CNN 对照，保持其他条件不变，再考虑对比学习或 episodic 训练。

## 四、随后执行：最小类别增量实验

先在 C4 内做一个清晰的 4+1+1 协议；不要一次把类别增量与跨分辨率变化混合。

| Session | Encoder 可用数据 | 分类类别集合 | 新增标注 |
| --- | --- | --- | --- |
| Base | 只来自四个 base 类的训练数据 | 四个 base 类 | base 训练预算完整记录 |
| 1 | encoder 冻结 | base 四类 + 第一 novel 类 | novel 类 K 条真实 support |
| 2 | encoder 冻结 | 全六类 | 第二 novel 类 K 条真实 support |

硬约束：

- 必须新训只见过 base 类的 encoder。两个 novel 类不能参与监督训练、预处理统计拟合、PCA、自监督预训练或超参数选择，否则需要另定义“见过无标注新类”的协议。
- 在实验开始前为每类固定 query pool，query 永不进入原型、协方差和训练。session 1 的同一批旧类 query 应在 session 2 继续测试，便于计算加入新类前后的性能变化。
- 新类 support 抽取一次后在后续 session 保留，不重新挑选更好的 support；未来类在其到来前不可见。
- base 类的类别库可采用训练集统计，但不得把 query 放进去。base 使用更多数据而 novel 仅 K-shot，可能产生旧新类偏置，应明确记录和评估。若采用固定大小旧类 exemplar 库，要为所有方法给相同的历史数据访问权限。
- 超参数在 base validation 或 base 类构成的开发 episodes 上选择，不使用最终两个 novel 类 query 选择收缩权重、网络或温度。
- 不能只选择一组容易的新类。初步试跑后，预先固定多种 base/novel 组合及 novel 到来顺序；六类数据的结论范围有限，后续再用更多类别的外部数据验证。

每个 session 输出：六类或当期全部已见类别的 CM、整体 Macro-F1/balanced accuracy、base 类性能、历史 novel 类性能、当前新增类性能、同一 query pool 的旧类性能变化。注明 base/old/new 的定义，避免在不同 session 中混用。

方法先保留 raw/PCA 模板、embedding Prototype、共享 Mahalanobis、每类 Mahalanobis 和 SVM 重新拟合基线。SVM 若要保留旧类，需要有旧类数据/特征访问；相同 memory 权限下比较，不能允许一方使用全部历史样本而另一方只能保留均值。

## 五、将 Adaptive Mahalanobis 放回增量目标

当前每类 support 协方差在低 shot 下估计不稳；可研究以下待验证形式：

\[
\widetilde\Sigma_c=(1-\alpha_K)\Sigma_{base}+\alpha_K\widehat\Sigma_c+\epsilon I.
\]

Sigma_base 仅由 base 训练集的类内残差估计，并在冻结时固定。alpha_K 只通过允许的开发数据确定。1-shot 时依赖 base 先验，5/10/20-shot 时逐渐允许新类别统计发挥作用，不能预先声称权重一定应采用某个增长规律。

这区别于当前“完全从单个 episode support 计算独立协方差”的做法；论文须作为新变体单独标注并消融。固定 base prior 也让添加新类时不必改变所有旧类的度量。若改用动态全类共享协方差，则必须计入对旧类分数的影响和重新计算成本。

## 六、最后增加域变化与成本评估

类别增量协议稳定后先复现到 C8；再做 C4→C8、C8→C4。同一来源的对应实物不得跨训练与 query。C4/C8 行号只是导出编号，不能直接当作跨文件物理样本配对证据。不能确认对应关系时，结论只能限于该数据划分，不能称严格独立的跨设备验证。

独立数据集、仪器、批次需要来源和元数据支持。统一为 2000 点只解决输入维度/波数位置，不证明仪器、噪声、样本差异已消除。

成本测量：分别记录 support 预处理和 encoder 前向、类别库更新、SVM 重新拟合、单谱/批量 query 推理、模型和类别库大小、保存旧样本数量。使用相同设备、重复计时、包含暖机；不要只比较分类头某一次耗时，也不要将源域预训练成本隐藏。

近期三个交付物应是：冻结特征诊断表、无 novel 泄漏的 4+1+1 协议及其数据清单、session 曲线与累计 CM。它们比立即扩大损失函数或添加网络数量更贴近研究目标。
