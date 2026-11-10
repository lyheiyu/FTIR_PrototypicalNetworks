# 最终 ProtoNet 消融：结果与边界

**核心结论：不能把普通 ProtoNet 训练写成新类识别改进。** 欧氏原型头固定时，EP 在六设置的 genuine novel Accuracy 均值全部低于 CE（A -4.83, B -6.02, C -7.74, D -2.91, E -9.18, F -9.85 pp）。A/B 总体准确率提高主要来自基类；不能用四个基类主导的总体分数替代新类泛化证据。

**有条件的正结果：** EP+收缩相对 CE+收缩，C/D 总体分别提高 +3.16/+3.20 pp，两方向各7/9个配对总体均值提高；但相对 PCA64+收缩，仍低 8.85/9.63 pp，六个设置总体均值均未超过 PCA。

**完整马氏对照解释了取舍：** 在同一 EP 编码器上，C/D 收缩相对 full 的 novel Accuracy 提高 12.86/10.73 pp，总体却变化 -0.59/-1.96 pp。收缩的 base→novel 错误率13.98%/19.12%，高于 full 的2.20%/3.51%；提升新类召回不等于改善全体增量性能。目标对齐基类库使 EP+收缩恢复14.00/17.44 pp，但额外使用目标基类标签。

**表征尺度限制：** CE C4/C8 训练特征范数110.79/105.58，EP为10.52/8.42；固定 ridge .001 在 EP 空间相对更强。这里评价固定训练和度量方案，不能宣称分离了尺度、抽样和损失各自的因果作用。没有根据结果重新缩放或调参。

## 预先固定的比较

18 个 episodic M1（每个编码器 47,400 参数），复用 18 个 CE M1，无 CE 重训。相同初始化、物理网格、原生 airPLS/归一化/三次插值顺序、保守记录重复组划分、4+1+1 到达顺序和评估 episode。欧氏 ProtoNet 只使用四个 base 类，以 4-way 5-support 10-query 训练，Adam .001，60 epochs；每 task 的更新数为 780/960/840。两类 genuine novel 不进入训练或源验证。

验证每 epoch 用固定20个源验证 episode（4-way 5-support 4-query），按最小查询 CE 选 checkpoint。验证 PS 最少9组，因此不借用训练数据或最终查询。训练支持/查询无交集。CE 与 EP 优化器更新数相同，但 CE 每步最多64监督标签、EP每步40查询标签且支持集参与梯度；EP按类平衡采样，CE为按行遍历，训练样本暴露和选择目标不完全相同。因此这是固定推断头下的 episodic 训练方案对照，无法进一步把原型损失与平衡采样各自贡献剥离。没有扫描超参数。

A=C4同域；B=C8同域；C=C4编码器/基类库，C8新增支持/查询；D=反向。E/F额外使用目标域每基类20条有标签参考，属于额外数据条件。均冻结 source-base-train 残差协方差先验；不更新旧类状态，不重训编码器。

## 主要结果（%）

| setting | CE_euclidean | EP_euclidean | CE_full | EP_full | CE_shrinkage | EP_shrinkage | PCA64_shrinkage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 80.825 | 88.872 | 90.892 | 92.227 | 95.149 | 96.525 | 97.265 |
| B | 78.557 | 87.541 | 90.922 | 91.410 | 95.844 | 96.027 | 97.845 |
| C | 78.047 | 72.185 | 81.902 | 80.992 | 77.246 | 80.403 | 89.251 |
| D | 66.683 | 68.679 | 82.988 | 80.442 | 75.282 | 78.481 | 88.112 |
| E | 77.079 | 81.389 | 90.879 | 90.015 | 94.428 | 94.403 | 96.981 |
| F | 78.080 | 77.197 | 90.617 | 91.303 | 94.678 | 95.916 | 96.883 |


Combined novel Accuracy:

| setting | CE_euclidean | EP_euclidean | CE_full | EP_full | CE_shrinkage | EP_shrinkage | PCA64_shrinkage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 74.225 | 69.394 | 76.726 | 77.382 | 90.860 | 91.346 | 93.072 |
| B | 75.428 | 69.411 | 77.795 | 78.861 | 94.118 | 93.430 | 96.777 |
| C | 75.147 | 67.409 | 81.713 | 82.588 | 94.587 | 95.444 | 96.428 |
| D | 74.322 | 71.411 | 77.907 | 81.404 | 91.314 | 92.137 | 93.404 |
| E | 75.570 | 66.395 | 78.265 | 77.744 | 92.596 | 91.283 | 95.505 |
| F | 73.228 | 63.380 | 76.838 | 76.911 | 89.048 | 89.759 | 92.100 |


Macro-F1:

| setting | CE_euclidean | EP_euclidean | CE_full | EP_full | CE_shrinkage | EP_shrinkage | PCA64_shrinkage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 79.891 | 87.778 | 90.209 | 91.609 | 95.127 | 96.456 | 97.204 |
| B | 77.123 | 86.579 | 90.523 | 91.027 | 95.854 | 95.984 | 97.823 |
| C | 76.575 | 70.689 | 81.720 | 81.037 | 76.178 | 79.879 | 88.975 |
| D | 64.047 | 67.542 | 82.486 | 79.792 | 73.216 | 77.222 | 87.976 |
| E | 75.225 | 79.947 | 90.490 | 89.656 | 94.419 | 94.358 | 96.952 |
| F | 76.672 | 76.136 | 89.915 | 90.736 | 94.560 | 95.805 | 96.796 |


## 基类误判、旧新类保持和关键混淆

以下均为主5/10-shot最终 session，Accuracy/错误率用%，previous_novel_change_pp 为该第一新类从 session1 到 session2 的变化（负值表示下降）。

| setting | method | base_accuracy | previous_novel_accuracy | current_novel_accuracy | base_to_novel_error_rate | previous_novel_change_pp |
| --- | --- | --- | --- | --- | --- | --- |
| A | CE_shrinkage | 97.293 | 90.272 | 91.448 | 2.478 | -9.256 |
| A | EP_shrinkage | 99.115 | 91.276 | 91.417 | 0.862 | -6.341 |
| A | PCA64_shrinkage | 99.361 | 94.502 | 91.643 | 0.465 | -4.256 |
| B | CE_shrinkage | 96.707 | 92.774 | 95.461 | 2.479 | -5.789 |
| B | EP_shrinkage | 97.326 | 92.854 | 94.006 | 1.260 | -3.415 |
| B | PCA64_shrinkage | 98.379 | 95.728 | 97.826 | 0.767 | -2.578 |
| C | CE_shrinkage | 68.575 | 91.909 | 97.265 | 18.143 | -7.135 |
| C | EP_shrinkage | 72.883 | 95.067 | 95.820 | 13.984 | -3.431 |
| C | PCA64_shrinkage | 85.662 | 95.783 | 97.072 | 7.822 | -3.372 |
| D | CE_shrinkage | 67.267 | 91.181 | 91.446 | 23.266 | -8.420 |
| D | EP_shrinkage | 71.652 | 92.306 | 91.969 | 19.124 | -5.946 |
| D | PCA64_shrinkage | 85.466 | 94.507 | 92.300 | 7.519 | -4.909 |


对应 genuine novel holdout 任务内，两方向平均互相误判率（%）：

| setting | method | HDPE_LDPE | PP_PS |
| --- | --- | --- | --- |
| A | CE_shrinkage | 12.064 | 10.672 |
| A | EP_shrinkage | 8.847 | 8.581 |
| A | PCA64_shrinkage | 4.353 | 9.314 |
| B | CE_shrinkage | 5.947 | 6.469 |
| B | EP_shrinkage | 6.944 | 3.036 |
| B | PCA64_shrinkage | 0.956 | 4.611 |
| C | CE_shrinkage | 8.458 | 4.878 |
| C | EP_shrinkage | 7.136 | 2.222 |
| C | PCA64_shrinkage | 3.361 | 4.181 |
| D | CE_shrinkage | 11.006 | 10.986 |
| D | EP_shrinkage | 10.208 | 8.331 |
| D | PCA64_shrinkage | 4.169 | 10.158 |


## 五个问题

1. **真正 episodic 训练是否改善新类识别？** 固定欧氏原型头，EP−CE 的新类 Accuracy 差值为：A -4.83, B -6.02, C -7.74, D -2.91, E -9.18, F -9.85 pp；总体 Accuracy 差值为：A +8.05, B +8.98, C -5.86, D +2.00, E +4.31, F -0.88 pp。必须同时看任务/种子配对差值，不能把换头带来的变化归因于训练方式。
2. **是否改善跨分辨率迁移？** C/D 固定头的实际目标表现和下表的迁移下降分别回答“目标效果”与“源到目标退化”。EP−CE 总体：欧氏 A +8.05, B +8.98, C -5.86, D +2.00, E +4.31, F -0.88；完整马氏 A +1.34, B +0.49, C -0.91, D -2.55, E -0.86, F +0.69；收缩马氏 A +1.38, B +0.18, C +3.16, D +3.20, E -0.03, F +1.24。迁移 gap 变小并不自动表示目标准确率变高，也可能源域表现下降。

| direction | head | CE_gap_pp | EP_gap_pp | EP_minus_CE_gap_pp |
| --- | --- | --- | --- | --- |
| C4_to_C8_gap | euclidean | -2.778 | -16.687 | -13.909 |
| C4_to_C8_gap | full | -8.990 | -11.236 | -2.245 |
| C4_to_C8_gap | shrinkage | -17.903 | -16.122 | 1.781 |
| C8_to_C4_gap | euclidean | -11.873 | -18.862 | -6.989 |
| C8_to_C4_gap | full | -7.933 | -10.969 | -3.035 |
| C8_to_C4_gap | shrinkage | -20.561 | -17.547 | 3.015 |


3. **收缩是否改善 episodic ProtoNet？** 同一 EP 编码器，收缩−完整马氏：A +4.30, B +4.62, C -0.59, D -1.96, E +4.39, F +4.61 pp；收缩−欧氏：A +7.65, B +8.49, C +8.22, D +9.80, E +13.01, F +18.72 pp。保留相同 .001 ridge、tau=5 和协方差 /n。不同训练目标会改变特征尺度，因此相同数值 ridge 并不保证相同相对正则强度，详见 feature_geometry.md。
4. **是否超过 PCA64 + 收缩？** EP 收缩−PCA 收缩：A -0.74, B -1.82, C -8.85, D -9.63, E -2.58, F -0.97 pp。该对照不能省略，也不能只报告最有利设置。
5. **最可辩护的贡献是什么？** 用可复用冻结表示和仅新增类注册构建可审计的 FTIR 4+1+1 框架；通过同编码器/同头和训练目标配对消融，界定 prototype 训练、协方差收缩及支持库分辨率匹配各自作用。只有对应配对结果支持的条件才能主张提升，不能将网络表示或收缩头表述为普遍优于光谱/PCA基线。

## 成本与统计口径

| objective | training_seconds | inference_median_ms_per_spectrum | optimizer_updates |
| --- | --- | --- | --- |
| CE | 77.616 | 0.279 | 860.000 |
| EP | 71.938 | 0.277 | 860.000 |


训练时间包括图编译、源验证和 checkpoint 写入；推断为 batch64、20次、CPU TF4线程，表中为每模型中位数再平均。历史 CE 与本轮 EP 顺序运行，壁钟时间仅作本机描述，非严格硬件基准。head 注册/预测时间不含光谱预处理与编码，见 costs.csv。仅注册当前类，无旧类重拟合；这里只比较原型头，无 SVM 重拟合。

Primary: final 6-class session, mean of 5/10 shots and 200 matched episodes within each task/seed, then mean over 3 tasks x 3 seeds. All deltas are percentage points. Tasks share data; seeds and repeated episodes are not independent datasets or specimens. No significance claim is made. Group Macro-F1 averages global per-class F1, including cross-group false positives.

全部四种 shot 和三个 session 见 metrics.csv / summary_by_shot_session.csv；基类、旧新类、当前新类、合并新类的 Accuracy/Macro-F1 均在原始 metrics.csv。主要对比表选取 Accuracy；详细 Macro-F1 需查汇总。类别召回见 per_class_metrics.csv，HDPE/LDPE 与 PP/PS 见 key_polymer_confusions.csv，旧新类保持与 base→novel 错误见 paired_differences.csv，CM 为按类别归一化的池化重复计数，不代表独立样本数。

## 限制与停止条件

保守记录/重复组隔离已审计，物理实物身份未验证；两库结果此前已检查，当前划分不构成未触碰的独立确证测试。C4/C8 不代表独立仪器；吸光度/透过率单位未获原始来源确认。相同原生 airPLS 索引平滑参数不等于相同物理平滑宽度。新旧结果均保留。无新架构、附加损失、增强或后验调参，本消融后停止扩展实验。
