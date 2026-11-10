# C4/C8 固定 encoder、配对 few-shot 比较

本入口以 MCCV2 的每类协方差方法为核心。旧实验文件不变，不加载训练范围不明的 H5。第一阶段只做 C4、C8 各自内部的已知类别少样本适配，不宣称未见类别或跨仪器泛化。

## 环境

在项目根目录、已有 TensorFlow 环境中运行。需要 numpy、pandas、scipy、scikit-learn、matplotlib、tensorflow；不需要 torch。现有 requirements.txt 中的相关依赖可用于准备环境，不必为了本实验安装 cadquery/Flask 等无关包。

```powershell
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

`python` 应指向你自己的 TensorFlow 环境；若未加入 PATH，用该环境 python.exe 的绝对路径替代，例如 PowerShell 中 `& '实际路径\python.exe' fewshot_benchmark.py ...`。

已在本机 `C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe` 验证七项检查（含一轮合成数据训练、checkpoint 保存和恢复、评价及出图）。该环境为 TensorFlow 2.13.0、NumPy 1.23.5、SciPy 1.10.1、scikit-learn 1.3.2、Matplotlib 3.7.5。尚未启动真实 C4/C8 的正式训练。

PowerShell 也可以直接指定这个环境，免去激活步骤：

```powershell
$py = 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe'
& $py fewshot_benchmark.py --output results/fewshot_smoke --datasets c4 --seeds 0 --shots 5 --epochs 2 --episodes 5 --val-episodes 3
```

## 先检查流程

去重文件已经生成；若以后重新生成，运行 `python deduplicate_plastics500.py`。

```powershell
python fewshot_benchmark.py --output results/fewshot_smoke --datasets c4 --seeds 0 --shots 5 --epochs 2 --episodes 5 --val-episodes 3
```

这会对 C4 做完整预处理，但仅训练两轮、评估五次；只用于检查依赖、训练和导出是否正常，不能作为论文结果。默认保留原代码的逐谱反转与 airPLS；如已确认原始物理量不需要反转，加 `--no-flip`。预处理选择应先确定，再统一用于所有方法。

## 正式运行

```powershell
python fewshot_benchmark.py --output results/fewshot_main --datasets c4 c8 --seeds 0 1 2 --shots 1 5 10 20 --epochs 60 --episodes 200 --val-episodes 30
```

输出目录必须不存在，避免覆盖结果；重跑时换新名字。长任务没有自动断点续训；已完成的各 seed 指标和模型会逐步落盘，预处理缓存保存在对应输出目录。该代码会在同一进程按数据集和 seed 顺序执行。

如果第一轮耗时过长，可先运行单个数据集；正式对比仍需保持相同配置。建议稳定后扩展到五个 seed。CPU 也可以运行；TensorFlow 是否能用 GPU 取决于你的环境。

## 实验定义

- 各数据集、各 seed 单独做分层 60%/20%/20% train/validation/test。
- train 的六类标注数据用于监督训练 encoder；validation 用于 EarlyStopping 与正则强度选择。encoder 架构为小型 Conv1D 分类网络，取 128 维倒数第二层特征。保留 MCCV2 的“监督预训练后冻结 encoder”路线，但不是原 H5 网络的逐层复现。
- test 从未参与 encoder 训练或正则选择。每个 episode 在 test 内每类取 K 条真实 support、15 条 query；两者不重叠。所有方法复用同一组索引。
- 比较 Euclidean、类内残差共享协方差、每类对角协方差、每类完整协方差（你的 class-adaptive 方法）。全部共享同一 encoder。
- 协方差统一按真实 support 数量取平均；完整马氏距离包含非对角项，只加入一次 ridge。采用 Woodbury 计算减少高维小样本求逆开销，数学上等同于完整二次型。
- 每种非欧氏方法、每个 K 在相同的 validation episodes 上从 1e-4、1e-3、1e-2 选择 Macro-F1 最好的 ridge；不在 test 调参。欧氏不调 ridge。
- 另含 raw-spectrum Linear SVM(C=1) 与 3-NN，它们使用相同真实 support/query，但不使用预训练特征。因此属于实用基线，训练数据预算与预训练方法不同；不能将该差异全部解释为距离收益。可用 `--no-raw-baselines` 关闭。
- 该协议的 K-shot 是“测试 episode 的 support 标注数”，不是整个项目仅使用 K 条/类标注训练 encoder。
- 各文件只做独立去重；第一阶段不跨 C4/C8 训练测试，因此暂不推断跨分辨率实物对应关系。
- 1-shot 时，当前 support 内的类内协方差为零，各正则马氏方法应退化为欧氏的相同分类。这不是新的自适应学习机制。

## 评价和出图

沿用 `utils.py` 中 `utils.printScore(y_true, y_pred)`，顺序为 Kappa、Macro-F1、Accuracy、Macro-Precision、Macro-Recall。新增命名指标、固定类别顺序的分类报告，以及非交互式 CM 绘制函数；旧 `plot_confusion_matrix` 接口保留。

每个 seed/shot 保存逐 episode 的总体和各类 precision/recall/F1、预测标签、真实标签、support/query 索引及累计 CM。split_manifest.csv 将索引对应回去重 CSV 保留的原 ID。每个 seed 保存 encoder.h5、embedding、训练曲线、正则选择结果。

完成后自动生成：

- `figures/shot_performance.png` 和 PDF：C4/C8 的 Accuracy 与 Macro-F1 随 shot 变化。
- `figures/c4_5shot_confusion_matrices.png` 等：每个数据集、每个 shot、各方法的 6×6 CM。行是真实类别、列是预测类别；逐行归一化，统一 0–1 色阶。
- 同目录中每个 CM 的 counts.csv 与 normalized.csv。
- `figures/per_class_recall_5shot.png` 和 PDF：各材料类别 recall 对比。
- `summary_metrics.csv`：seed 均值、seed 标准差和均值的 95% t 区间。
- `per_class_metrics.csv`：按全部 query 出现次数汇总的各类指标。
- `source/` 和 `provenance.json`：本次源代码快照、依赖版本、数据 SHA-256。

反复抽样的 episode 会重复使用 query。累计 CM 的计数表示 query 出现次数，不是独立实物数；图中误差条按各 seed 的平均指标计算，描述随机划分、训练和抽样的联合变化，不是独立实物总体的置信区间。只有一个 seed 时不估计该区间。

已有结果重新出图，无需重新训练，也无需 TensorFlow：

```powershell
python plot_fewshot_benchmark.py results/fewshot_main
```

只在完整运行结束后出总图；程序会检查总体指标记录是否齐全。

## 最小验证

```powershell
python -m unittest discover -s tests -p "test_fewshot_benchmark.py" -v
```

覆盖完整协方差二次型、1-shot 退化、非对角项作用、数据划分隔离、episode 配对、utils 指标方向和合成数据的导出/绘图流程。合成数据测试仅用于检查程序，不是正式实验。
