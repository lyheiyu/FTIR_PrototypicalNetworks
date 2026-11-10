# 可复用 FTIR Encoder 研究流程：运行指南

入口为 `run_ftir_research.py`，有 `inventory`、`diagnose`、`incremental`、`plot` 四个子命令。

本版实现研究计划中可执行的主线：现有特征诊断、基础类专用 encoder、冻结后的连续类别加入、基于 base 先验的协方差收缩、旧新类联合评价、跨文件目标域适配、逐类 CM 与结果保存。自监督/对比/重建联合损失仍是研究候选，没有将其作为已经验证的功能加入。先用诊断决定是否需要这类训练目标。

## 1. 环境和数据审计

项目根目录，已激活 fewshotlearning 的 PowerShell：

```powershell
python run_ftir_research.py --help
```

未激活时，用 `& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe'` 替代每条命令开头的 `python`。

本次已经生成 `results/research_inventory/dataset_classes.csv` 和每个文件的 JSON/逐行处理记录，不必重复运行。更新源数据后可运行以下命令，输出目录需换为尚不存在的名称：

```powershell
python run_ftir_research.py inventory --output results/research_inventory_v2
```

实际审计结果（逐谱数值严格去重、无效行剔除后）：

| key | 文件/用途 | 类别数 | 有效不同光谱 |
| --- | --- | ---: | ---: |
| c4 | 去重 C4，主实验 | 6 | 2649 |
| c8 | 去重 C8，主实验复现 | 6 | 2390 |
| d4 | D4_4_publication11.csv，类别扩展验证 | 11 | 954 |
| second | new_SecondDataset2.csv，候选外部域 | 4 | 786 |
| jung | 216… modified.csv | 12 | 168 |
| jung_v2 | 同来源的修改版本 | 12 | 168 |
| fourth_original | FourthDataset2.csv | 14 | 345 |
| flopp_r | FourthdatasetFollp-r3.csv，未补零裁剪版 | 14 | 179 |
| flopp_e | FourthdatasetFoppl-e.csv | 7 | 170 |

其他 fourth 版本也有适配器，详见 inventory。不要把这些变体合并后当作独立样本；名称相似也不能证明来源独立。FTIR_PLASTIC_c4/c8 原始元数据格式的有效数量与相应 500 版本相同，不能据此当作额外样本量。旧 CSV 可能含空白、非有限数值、重复谱或补零区，本版记录剔除/裁剪原因，不自动用零填充缺失值。

标签统一只处理明确名称，例如 Poly(propylene)→PP；PE、HDPE、LDPE 保持不同。Nylon 不自动合并为 PA，Polyester 不自动合并为 PET，PE+fouling、Morphotype、Cotton 等保留原类别含义。并非全部标签都代表纯聚合物物种，跨文件比较前需核对化学定义。

## 2. 下一步首先运行：冻结特征诊断

无需训练网络，复用 `results/fewshot_main` 的预处理、embedding、划分及 episode 索引。PCA 在原 encoder train split 拟合，分类器参数只在原 validation split 选择。

```powershell
python run_ftir_research.py diagnose --previous results/fewshot_main --output results/frozen_diagnosis_v1 --datasets c4 c8 --seeds 0 1 2 --shots 1 5 10 20 --episodes 200 --val-episodes 10
```

默认比较 raw SVM、raw 1-NN、PCA SVM、embedding SVM、embedding 1-NN、embedding Prototype、embedding 每类 Mahalanobis。可通过 `--methods` 增加 `raw_proto`、`pca_proto` 等。默认 PCA=64 维，可用 `--pca-dim` 配置；第一版保持这个预先规定值，不按 test 最好分数选维数。

原始 embedding 的 `embedding_class` 沿用旧实验选定的 ridge，保证距离结果可对照；若启用 `--l2-embedding` 则是一个新变换，ridge 重新在验证集选取。其余分类头按验证集选参数，选参记录落盘。Raw SVM 在新诊断中也会调 C，与旧实验固定 C=1 的结果应分别标注。

有一个已经完成的短流程检查 `results/research_diagnostic_check`：只用 C4/seed0/5-shot/3 episodes/PCA16，不能作为正式结论。它的 Adaptive Mahalanobis 预测已与旧实验前三个 episode 的 270 个 query 逐条核对，一致。

诊断回答当前已知类别的表征/分类规则问题，不证明对新类泛化。

## 3. 真正的类别增量：先小跑 C4 的 4+1+1

```powershell
python run_ftir_research.py incremental --dataset c4 --output results/c4_incremental_smoke --tasks 1 --seeds 0 --shots 5 --queries 15 --epochs 2 --repeats 2 --val-episodes 2
```

每个任务先选四个 base 类，只用它们的训练光谱训练 encoder；另外两类彻底不参与训练、PCA 和基础协方差估计。之后冻结 encoder，第一 session 加入一个新类，第二 session 再加入一个新类。每个 session 都识别所有已见类别。

重要协议：

- Query pool 每类在运行开始时固定；旧类 query 在后续 session 保持相同。
- Base validation 是独立保留的 base 类样本。每种 head 的 C/ridge/收缩 tau 只通过它选择，不使用 novel query。
- 若小数据集 base validation 不足 K+Q，验证 episode 的 K/Q 会自动缩小并在 `*validation.csv` 中明确记录；正式 novel support 的 K 和 query 数量不会缩小，不足会报错。
- 默认每个 base 类保留最多 20 个训练 exemplar；新类保留 K 个 support。所有 head 使用相同 exemplar 索引。Base 类多样本、新类少样本的不对称是本协议的一部分。
- 每个 repeat 内旧 support 随 session 保留，只添加新类 support；不从 query 更新原型。
- Encoder 推理时可对 novel 光谱生成 embedding，但没有反向训练或目标统计更新。
- 预处理按单条光谱完成；PCA 和共享先验只在 base train 拟合。每个 seed/task 的库和模型独立。
- `tasks.json` 保存任务定义；改变任务顺序和网络时需作为新实验标注。初始自动任务是随机开发任务，不保证类别组合全面覆盖。

正式 C4 运行示例：

```powershell
python run_ftir_research.py incremental --dataset c4 --output results/c4_incremental_v1 --tasks 3 --seeds 0 1 2 --shots 1 5 10 20 --epochs 60 --repeats 20 --val-episodes 10
```

先检查 C4 结果再复现 C8：

```powershell
python run_ftir_research.py incremental --dataset c8 --output results/c8_incremental_v1 --tasks 3 --seeds 0 1 2 --shots 1 5 10 20 --epochs 60 --repeats 20 --val-episodes 10
```

默认是上一轮 compact 监督 CNN。可用 `--backbone multiscale` 比较保留峰位置的多尺度监督 CNN；这是新的架构消融，不应同时改变其他条件后把提升都归因于分类器。可用 `--embedding-dim` 改维数。

输出目录必须不存在，不覆盖旧实验。本版没有断点续训；各 trial 的指标、模型会逐步保存，预处理缓存跨运行保存在 `results/research_cache`。运行中断时保留已有文件，换目录重新启动；未标记 COMPLETE 的结果不能当成完整正式结果。

## 4. Adaptive Mahalanobis 的增量版本

默认主比较：raw SVM、raw 1-NN、raw Prototype、PCA Prototype、embedding Prototype、embedding 共享基础协方差、embedding 每类协方差、embedding 收缩协方差。可用 `--methods` 调整。

这里的 `embedding_shared_prior` 与上一轮 support 内共享协方差不同：它使用仅由 base train 类内残差估计的固定 prior。它不会随着未来类别加入重新估计。

收缩方案：

\[
\widetilde\Sigma_c=\alpha_c\widehat\Sigma_c+(1-\alpha_c)\Sigma_{base}+\epsilon I,
\qquad \alpha_c=\frac{n_c-1}{n_c-1+\tau}.
\]

tau 和 ridge 只在 base validation 选择。1-shot 时 alpha=0，依赖 base prior；它不会像纯类别协方差那样必然退化为欧氏。这个公式是待评估的新变体，不保证优于原方案。所有 full 距离都保留非对角项，正则只加一次。

## 5. 其他数据集与候选跨域实验

D4 有更多类别，适合先验证增量流程是否仍有效：

```powershell
python run_ftir_research.py incremental --dataset d4 --output results/d4_incremental_v1 --tasks 3 --seeds 0 1 2 --shots 1 5 --queries 5 --epochs 60 --repeats 20
```

FLOPP-r/Jung 样本更少，不能照搬每类 20-shot+15 query。可先用较低预算：

```powershell
python run_ftir_research.py incremental --dataset flopp_r --output results/flopp_r_incremental_v1 --tasks 3 --seeds 0 1 2 --shots 1 5 --queries 3 --base-memory 5 --epochs 60 --repeats 20
```

代码会记录实际保留类及数量，不静默补造不足的样本。不要将不同材料标签自动归并以凑样本数。

可选跨文件模式示例：second 的 HDPE/LDPE/PP 作为 base，目标 C4 的 PET/PVC/PS 是新类。这里同时存在类别变化和候选域变化，要与同域实验分开解释：

```powershell
python run_ftir_research.py incremental --dataset second --target c4 --output results/second_to_c4_v1 --base-count 3 --novel-sessions 3 --tasks 3 --seeds 0 1 2 --shots 1 5 10 20 --queries 15 --epochs 60 --repeats 20
```

统一网格使用 source/target 的实际共同波数范围，无外推。文件默认谱反转设置参考现有读取器，尚不是对原始仪器单位的独立认证；应核实物理量后固定，可用 `--flip yes/no/profile` 控制，`--no-baseline` 是单独预处理消融。

跨文件不自动等同跨设备。数据来源、仪器和实物元数据未确认时，只能标为候选跨文件评估。C4→C8、fourth 各变体之间等同 family 的跨文件模式要求先提供核实的实物分组，不允许仅凭导出编号假定独立：

```text
dataset,row_id,group
c4,1,verified_specimen_001
c8,1,verified_specimen_001
...
```

使用 `--groups verified_groups.csv`，必须覆盖所选文件的全部保留行。同一实物每文件只使用一个代表，source 训练/验证排除 target query 的分组。Group ID 需来自你确认的元数据，程序不自动编造。跨 family 如未提供 groups，只控制文件内精确重复和可检测的跨文件精确重复，不能保证没有物理重复。

自定义任务可用 `--task-file`，避免只使用默认随机组合，例如仓库里的 `research_configs/c4_4plus1plus1.json`。文件中的类别必须匹配 inventory 输出的规范名称。

## 6. 指标、图与可追溯性

沿用 utils.py 五项指标并导出逐类 precision/recall/F1。每个 task/seed 目录保存：

- `split_manifest.csv`：原始 ID、规范类别、group、base_train/base_validation/fixed_query/novel_support_pool 角色。
- `training_provenance.json`：训练索引、基础类别、模型 hash、训练/整体特征提取时间、模型文件大小。
- `encoder.h5`、`pca.npz`、冻结特征和固定 base covariance。
- `sessions.json`：每次 support、query、已见类别及新增类别；`*example_library.npz` 为 repeat0 的最终类别 exemplar 库。
- `metrics.csv`：总体五项指标、基础类/旧类/当前新类 accuracy、旧类 accuracy 相对上一 session 的变化百分点。
- `per_class_metrics.csv`、`predictions.csv`、CM 数值及类别顺序。
- `FROZEN_CHECK.json`：encoder 文件在增量评估前后保持不变。训练只发生在 base 阶段，分类头评估无 encoder optimizer。

自动图片包括 shot 诊断曲线、session 性能曲线，以及每个 task/seed 的各 session CM，均保存 PNG/PDF。默认 CM 画 5-shot；若没有 5-shot，画第一个请求的 K，其他 K 的 CM 数值仍全部保存。不同任务的已见类别不同，不强行合并它们的 session CM。

```powershell
python run_ftir_research.py plot results/c4_incremental_v1
```

root 下 `trial_metrics.csv` 和 `summary_metrics.csv` 保存按 trial 聚合的均值/标准差。重复 episode、不同类别任务及重叠 seed 划分不是独立实物样本，不能把所有记录当成独立样本做显著性检验。

`fit_ms`/`predict_ms` 仅是 head 对相同 exemplar 的完整拟合及批量 query 推理；当前模拟器为清晰比较，会重建每个 head。它不是只更新单个新类的优化部署实现，不包含共享的光谱预处理/encoder 前向，不能直接据此宣称整体更新更快。`exemplar_bytes` 是对应特征库大小，不包括全部模型/PCA 参数和 Python 运行时。正式成本研究需要分项计时、暖机和相同硬件重复测量。

## 7. 验证与已有检查结果

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

新测试覆盖不同标签格式、缺失/重复行、保守标签映射、物理分组、基础类隔离、PCA 不使用 future 类、完整协方差及 1-shot prior、连续 session support/query 保留、encoder 训练和出图。

已完成全目录 inventory、短 C4 冻结诊断、合成数据增量测试，以及 D4 的单轮/低维短流程检查。`research_*_check` 均仅用于验证代码，不能用于论文性能结论；尚未启动正式诊断或正式增量训练。
