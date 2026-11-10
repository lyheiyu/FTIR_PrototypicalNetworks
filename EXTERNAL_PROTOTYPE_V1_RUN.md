# 外部 FTIR 与冻结原型诊断 V1

本轮不训练编码器；旧脚本和结果不改动。新增入口复用现有数据适配、分类头、指标及混淆矩阵函数。共享/收缩 Mahalanobis 的数学公式与旧 `Head` 一致，新实现使每次注册只处理到达类，保持旧类状态。

## 本轮实际范围

先审计 `dataset/D4_4_publication11.csv`。D4 为 959 行、11 类，去除五条完全重复记录后 954 条；覆盖 599.78354–3995.98555。冻结 M1 输入网格覆盖 399.1927–4000.6045，107 个坐标在 D4 实测范围之外。因此外部适配和 C4→D4 实验均停止，不外推、不补零、不修改网格。信号物理语义和实物身份也未核实。审计文件为 `results/external_transfer_prototype_v1/d4_dataset_audit.md`；外部结果 CSV 明确记录 blocked 和空指标。

C4 继续完整三任务、三种子、1/5/10/20-shot、每 shot 200 个配对 episode。直接读取 `frozen_incremental_v1` 的 `sessions.json`，包括基类 exemplar、support/query、类到达顺序。每类基类 20 条，新类 K 条，旧支持集保持不变。

新增光谱 1-NN、冻结 PCA64 1-NN 和 M1 的四种原型头，比较原 M1 1-NN/SVM、光谱 SVM。PCA 仅在原 base_train 拟合，64 维、无 whitening、randomized SVD、对应 seed；之后冻结。原型：

- Euclidean：原始 M1 特征的类均值，平方欧氏距离。
- Cosine：原始特征先求类均值，再对均值归一化，计算余弦距离；不是先逐样本归一化后求均值。
- Shared：原 base_train 的类内残差协方差 + 0.001 I，逆矩阵固定；各类只保存均值。
- Shrinkage：每类 `cov = residual.T @ residual / n`，`alpha=(n-1)/(n-1+5)`，`precision=inv(alpha*cov+(1-alpha)*base_prior+0.001*I)`。新类只计算自己的均值与矩阵，旧类不变。

这些是冻结特征的统计分类器，不是新训练的 ProtoNet。没有新损失、增强、查询驱动调参或新 checkpoint 选择。PCA/协方差有合法的基类统计访问阶段，原型均值只能用与 1-NN 相同的 exemplar/support。基类 query 已参与父实验模型选择，结果仍为探索性验证集复用，不能当独立基类测试。

## 命令

已完成的目录不必重跑。新运行需要新目录，下面是本轮实际流程：

```powershell
$ftirPython = 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe'
& $ftirPython audit_external_ftir_v1.py --output results/external_transfer_prototype_v1
& $ftirPython -m unittest discover -s tests -p test_external_prototype_v1.py -v
& $ftirPython run_external_prototype_v1.py --output results/external_transfer_prototype_v1/validation_smoke_checked --smoke
& $ftirPython run_external_prototype_v1.py --output results/external_transfer_prototype_v1 --validated-smoke results/external_transfer_prototype_v1/validation_smoke_checked
```

重新汇总、出图，不重新评估：

```powershell
& $ftirPython analyze_external_prototype_v1.py results/external_transfer_prototype_v1
```

中断后用原正式命令追加 `--resume`，复用已完成 trial；配置和源代码必须匹配。未完成 trial 从头评估。不允许通过 resume 悄悄改协议。修改代码需要重新验证并使用新输出目录；旧试运行保留。

## 成本与审计

原三个参考头复用完整父指标；在每 trial/shot/session 前 20 个相同 episode 重新拟合和计时，并逐项核对所得指标与父运行一致。所有新方法也取相同前 20 个 episode 做成本比较，避免混用父运行线程条件。NumPy/sklearn 数值线程为 1，TensorFlow 为 4。准确率仍使用完整 200 个 episode。

`new_feature_encoding_ms` 是新增 K 条的实际冻结编码或 PCA 投影。`head_update_ms` 为原型到达类注册或参考分类器全库准备/拟合；SVM 完整 refit 覆盖全部保留 exemplar，不是完整历史数据重放。`new_class_registration_ms` 包含特征变换和头更新，排除共享逐行预处理。`prediction_ms` 使用缓存 query 特征，不包含 query 编码。基类初始化、PCA、先验拟合成本另存，不能将局部注册时间当全部部署开销。

`stored_feature_bank_bytes`、`class_metric_bytes`、`global_metric_bytes`、`global_transform_bytes` 分列；矩阵和 PCA/CNN 变换参数都统计，Python/标签对象开销不计，不是进程 RAM 测量。共享 Mahalanobis 含一份 128×128 矩阵，收缩版本含每类精度矩阵和全局先验；不能只报告均值原型的字节。

完整结果在 `results_overview_zh.md`、`c4_baseline_comparison.csv`、`prototype_comparison.csv`、`per_class_metrics.csv`、`cost_comparison.csv`。`paired_differences_pp.csv` 与 `primary_paired_summary.csv` 给出配对差值；`novel_class_recall_summary.csv` 仅统计各类作为新类的 recall。所有 query 出现次数是重复 episode 计数，不是独立实物数量。

本轮完成后增加了人工审核的具体解读 `findings_zh.md`，并将其用作最终 `results_overview_zh.md`；详细自动表格另存 `detailed_metrics.md`。`shrinkage_vs_references_pp.csv` 补充收缩头与所有参考的直接配对差值。重新运行分析入口会重新生成自动 overview；独立解读、详细表格备份和补充配对文件仍保留，可直接阅读 `findings_zh.md`。`audit_protocol.py` 可在项目环境中再次检查父文件、模型、PCA 和先验来源，不会训练模型。
