# 冻结 M0/M1 的 C4 4+1+1 评估

检查旧实现后复用 `ftir_research_core.Head`、`score` 和 `utils.draw_confusion_matrix`。旧增量入口会重新生成划分和训练编码器，且 `Recorder.old_accuracy` 将所有旧类（包括基类）合在一起，也不提供用户要求的各组 Macro-F1。因此保留旧代码，新增薄适配入口 `run_frozen_incremental_v1.py` 读取架构实验产物；不调用训练，不新增编码器、损失或增强。

## 数据访问

源目录为 `results/encoder_architecture_v1`，必须是完整的三任务、三种子正式运行。保存的共同 1868 点物理网格、逐行预处理、C4 数据文件哈希、样本角色、M0/M1 checkpoint 和嵌入全部复用。加载后设置 `trainable=False`，校验加载权重与源运行哈希相同，评估前后再次检查权重与文件哈希。

每个任务/种子固定从原 `base_train` 中每类选 20 条 exemplar，与旧增量实验默认内存预算一致。M0 SVM/1-NN、M1 SVM/1-NN、光谱 SVM 使用完全相同的支持索引。基类 exemplar 在所有 shot 和 episode 中保持不变。

每个 K 的 200 个新类 episode 直接复用架构比较中保存的 support/query。Session 0 是四个基类；Session 1 只加入任务定义的第一个新类及其 K 条 support；Session 2 保留全部旧 support 和 query，再加入第二个新类。不存在未来类的提前访问。三个任务的新类顺序不改变。

每类 15 条 query。新类 query 原样复用。基类 query 从原 `base_validation` 采样，在同一个 episode 的三个 session 中不变，同一 episode 编号的各 shot 也保持一致。**原基类验证集已经用于 checkpoint 和 SVM C 选择，因此基类和总体指标是验证集复用的探索性估计，不能称为独立测试。** 在不改变原划分和重新训练的前提下，无法补出真正未参与选择的基类测试集。新类 support/query 仍来自原本互不重叠的两个池；目前没有实物级分组保证。

SVM 直接复用架构比较中对应 task/seed/shot/view 的基类验证最优 C，不用新类 query 调参。没有访问完整新类训练集，也没有额外把全量基类训练数据交给 SVM。本轮的“完整 SVM 重拟合”指在**所有已见、保留的 exemplar**上从头拟合线性 SVM：80、80+K、80+2K 行。它不是在完整历史原始数据库上重训的基线。

SVM 的历史最优 C 与 shot 相关，因此即使 Session 0 的 exemplar 不变，其 SVM 结果也可能随 shot 改变。M0/M1 的编码器此前使用了完整 base_train，光谱 SVM 没有这个表征学习阶段；本轮匹配的是增量分类头的 exemplar 预算，不能称为全部历史监督数据访问完全相同。详细条件见输出目录 `data_access_conditions.md`。

## 指标定义

每个 session 给出 base、previous_novel、current_novel、seen_novel、all 五组 accuracy 和 Macro-F1。previous_novel 排除基类和当前新类，直到 Session 2 才非空；seen_novel 包含当前类和此前引入的新类；Session 0 没有新类。空组记为 NaN。

分组 accuracy 是该组真实标签样本的正确率。分组 Macro-F1 是完整已见类别混淆矩阵中，该组各类 one-vs-rest F1 的平均，**包含来自其他组的 false positives**。不能先把预测过滤为该组后再计算。all 指标准确率和标准多类 Macro-F1。

平均增量准确率同时给出 Session 0/1/2 平均和仅 Session 1/2 平均。先按 episode 汇总到 task/seed，再对九组等权平均；重复 query 出现次数不能视为独立实物样本。

## 成本定义

- `new_class_encoding_ms`：预热后的冻结编码器对实际到达 K 条 support 的重执行时间；评分沿用已保存的嵌入，首个 episode 检查重新编码与缓存一致。光谱视图没有编码器。
- `local_bank_append_ms`：仅复制新类的 K 条特征到类库，保留旧类，不遍历旧类；它不是完整服务更新时间。
- `local_registration_ms`：上述两者相加，排除共享的逐行预处理。
- `memory_assembly_ms`：把已见类库整理为分类器输入，单独计时。
- `svm_full_refit_ms`：在整个已见 exemplar 库上重新训练 SVM。不能将新类入库时间冒充这个开销。
- `knn_index_build_ms`：复用旧 sklearn 1-NN 分类头时的索引准备成本，单独记录。
- `total_head_update_ms`：新类编码、入库、整理和 SVM 重拟合或 1-NN 索引准备的总和。
- `predict_ms`：缓存 query 特征上的分类器预测；Session 0/1/2 分别为 60/75/90 条 query，不包含 query 特征提取或预处理，不是端到端推理时间。

## 命令

PowerShell 在项目根目录运行：

```powershell
$ftirPython = 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe'
& $ftirPython -m unittest discover -s tests -p test_frozen_incremental_v1.py -v
& $ftirPython run_frozen_incremental_v1.py --output results/frozen_incremental_v1_smoke_checked --smoke
& $ftirPython run_frozen_incremental_v1.py --output results/frozen_incremental_v1 --validated-smoke results/frozen_incremental_v1_smoke_checked
```

现有目录默认不覆盖。已完成后不需要再训练或重新评估，仅重新汇总出图：

```powershell
& $ftirPython analyze_frozen_incremental_v1.py results/frozen_incremental_v1
```

中断后使用原运行命令追加 `--resume`；配置、源代码哈希必须一致，复用已完成 trial，未完成的 trial 从头评估。若修改实验代码，需要新的 smoke 和正式输出目录。首次小测试失败的导出目录也保留，正式结果以 `frozen_incremental_v1` 中的 `COMPLETE.json` 为准。

## 输出

`incremental_analysis.md`、`conclusion.md`、`cost_analysis.md`；完整指标 `metrics.csv`；各组每 session 九组平均 `summary_metrics.csv`；每类 precision/recall/F1 `per_class_metrics.csv`；逐任务/种子结果 `trial_metrics.csv`；平均增量指标 `average_incremental_metrics.csv`；成本明细与汇总；源代码快照；各 trial 的 `sessions.json`、`input_provenance.json`、`frozen_check.json`；`protocol_audit.json`。`figures` 包含各 shot 的 session 曲线和各任务 5/20-shot 最终六类混淆矩阵，均保存 PNG/PDF，混淆矩阵另有原始计数 CSV。
