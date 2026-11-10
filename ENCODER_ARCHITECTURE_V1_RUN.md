# M0/M1 frozen representation comparison V1

本实验只比较两个编码器，不增加损失、数据增强或其他模型。输出保存在新目录，旧实验不变。

## 运行

在项目根目录的 PowerShell 中运行。可直接指定解释器，无需先激活 conda：

```powershell
$ftirPython = 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe'
& $ftirPython -m unittest discover -s tests -p test_encoder_architecture_v1.py -v
& $ftirPython run_encoder_architecture_v1.py --output results/encoder_architecture_v1_smoke --smoke
& $ftirPython run_encoder_architecture_v1.py --output results/encoder_architecture_v1 --validated-smoke results/encoder_architecture_v1_smoke
```

已经存在的输出目录不会被静默覆盖。中断后，使用原命令追加 `--resume`，复用已完成的模型和配对任务；未完成的单个模型会从头训练。配置、数据和代码哈希必须与原运行一致。修改实验代码后，需要新目录并重新做 smoke 验证。

仅重新汇总及出图：

```powershell
& $ftirPython analyze_encoder_architecture_v1.py results/encoder_architecture_v1
```

## 固定协议

- 数据：去除完全相同光谱的 C4。C8 只参与确定共同物理波数范围，没有用于训练或独立确认。
- 输入：共同范围内的 1868 个升序坐标；逐行 `max-y → native-grid airPLS → maxabs → cubic`。保存完整坐标、源数据及代码哈希。该强度变换不等同于已经确认的物理吸光度单位。
- 三个任务：分别留出 HDPE/LDPE、PP/PS、PET/PVC；每个任务只用另外四类训练编码器。随机种子 0、1、2。
- 基类每类按 80/20 划分训练/验证；新类每类按 50/50 划分互不重叠的 support/query 池。每个 shot 使用 200 个共享的两类 episode，每类 15 条 query。
- 1、5、10、20-shot；同一任务、种子下，所有分类器使用完全相同的 episode。
- Adam、学习率 0.001、batch 64、60 epochs、相同的逐 epoch 样本顺序；仅基类交叉熵。固定训练预算后，选择基类验证 loss 最低的 checkpoint。
- 冻结后比较 M0/M1 的线性 SVM 和欧氏 1-NN；预处理光谱的线性 SVM 为参考。SVM 的 C 只通过基类验证 episode 在 0.1、1、10 中选择。
- 两个编码器输出均为 128 维，直接评价未额外归一化的嵌入。

M0 为紧凑 CNN，47,872 个编码器参数。M1 使用 9/25/49 点多尺度卷积、较少池化和 32 个保留顺序的局部谱段聚合，47,400 个编码器参数，约少 0.99%。两者的四类训练头都另外增加 516 个参数。该实验比较完整 M1 设计，不能单独归因于位置保留或多尺度中的某一项。

## 预先确定的停止标准

主指标是 5/10-shot 准确率的等权平均。SVM 和 1-NN 各自都必须满足：

1. M1 比 M0 平均至少高 1 个百分点。
2. 九个任务/种子配对中至少六个有提升。
3. 三个任务各自平均提升均为正。

任一分类器未满足，就停止本次架构调查，不追加模型或损失来挽救结果。这是描述性的决策标准，不是显著性检验。

## 输出及解释

`architecture_comparison.md` 和 `accuracy_table.csv` 给出准确率及模型成本；`macro_f1_table.csv` 给出 Macro-F1；`paired_deltas.csv` 保存逐任务/种子的差值；`conclusion.md` 给出停止判断。`figures` 保存曲线及按留出类别命名的混淆矩阵，PNG/PDF 均可用于检查。混淆矩阵累计的是重复 episode 中的 query 出现次数，不是互相独立的新实物数量。

`costs.csv` 记录每个模型的实际训练秒数、参数量、固定更新步数和特征提取成本；推理成本为预热后的 batch-64 编码器推理时间，按每条光谱折算，不包含预处理和分类器。`metrics.csv` 另存 SVM/1-NN 的支持集拟合及 query 预测时间。不同硬件上的耗时不能直接比较。

每个任务/种子目录中的 `M0/encoder.keras`、`M1/encoder.keras` 是已选定的编码器。单独加载 M1 时，先 `import encoder_architecture_models` 注册位置聚合层，再调用 `tf.keras.models.load_model(path, compile=False)`。新输入必须沿用此次保存的完整预处理与 `common_grid.npy`，不能只按长度截取光谱。

这是新类两类识别实验，不能当作六类混合识别或 4+1+1 增量识别的结果。C4 新类之前已被查看；目前仅审计完全重复光谱，尚未建立实物级分组，因此全部结果为探索性证据。
