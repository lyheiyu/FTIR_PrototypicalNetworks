# v2 研究计划的可实施性审查

审查对象：FTIR_Reusable_Encoder_FSCIL_Detailed_Research_Plan_v2.md。
结论：主线在现有环境中可实现，但不宜一次展开全部工作包。当前最需要补充的是物理信号语义、固定且不泄漏的开发协议、多方法 harmonizer 和持久 Feature Bank，而非立即增加 Encoder 数量。工程可运行不等于性能、跨域能力或论文新颖性已经成立。

本次只审查和记录，没有修改原计划、历史实验代码或正式结果，也没有训练新模型。

## 1. 相比上一版的实质变化

- 插值方法由固定前提变为实验变量；目标网格独立于插值方法比较。
- 输入兼容、谱形保真与下游泛化分开验收。
- 补充真实透过率/吸光度语义、分辨率与采样间隔、历史和严格分组协议。
- 明确新预处理必须配套新训练的 Encoder，不能用旧权重的输入分布不匹配来判断插值好坏。
- 从实验脚本进一步要求持久、版本兼容、只更新新类的 Feature Bank。

这些变化可沿用现有结果和代码作为 historical/legacy 分支，不需要推倒重写。

## 2. 当前代码与 v2 工作包的对应

| 工作项 | 当前状态 | v2 所需增量 |
| --- | --- | --- |
| 本地多文件适配、数值去重、逐行审计 | 已有 ftir_research_data.py、research_inventory | 信号单位、真实 specimen/device/batch 元数据、近重复候选和完整历史 manifest |
| 历史冻结特征诊断 | run_ftir_research.py diagnose 可运行，已有短流程验证 | 补历史 shared-support 度量、cosine、RBF 候选和明确的 checkpoint 兼容清单 |
| 通用 CSV | 只有本地文件布局的显式 registry | 新增两列竖表、共轴多谱列、首行波数宽表以及 schema hints；不能承诺任意未知 CSV 自动正确识别 |
| Harmonization | 固定 interp1d(kind='cubic') | linear、legacy cubic、PCHIP、粗网格聚合/抗混叠；统一返回真实物理网格和 QC |
| 光谱保真 / 方法×网格 SVM | 未实现 | 原生网格对照、遮点重建、峰指标、同 manifest 的方法×网格实验 |
| Base-only 训练及 4+1+1 | 已实现，合成数据和 D4 短流程通过 | 输入协议锁定后进行正式训练；开发/最终 novel 类的边界需更明确 |
| Encoder 训练目标 | 当前新入口仅 CE；compact/multiscale 两种结构 | 修复/统一的 episodic，以及可选 SupCon/组合消融 |
| 每类/共享/收缩协方差 | 已实现 | 文档公式和统计定义与代码对齐；不是全部需要重新写 |
| 持久 Feature Bank | 有最终 exemplar 库快照和 session 记录 | 真正 add/update/predict/save/load、版本检查、只增新类统计、内存中权重/状态检查 |
| FSCIL 指标与 CM | overall/base/old/current-new、Macro-F1、逐类 CM、相邻旧类变化已实现 | AIA、历史最佳定义的 forgetting、old↔new confusion、真正局部更新成本 |
| 跨文件模式 | 已有 source→target 联合类别/域变化模式 | 补 G1a 固定源原型的直接迁移；C4/C8 的严格物理分组依赖元数据 |

当前实验不是通用自动 CSV 系统，也不是已经完成 v2 的所有模块。

## 3. 必须先修正的细节

### 3.1 历史强基线并非 native/raw，也不是 RBF

当前 sv​​m_raw 的输入已经做逐谱反转、airPLS、max-abs 归一化和共同 2000 点 cubic 插值。它应标为“legacy-preprocessed spectrum + linear SVM”，不能把已有 95–98% 结果写为未经处理的原生网格 RBF SVM。

v2 新增 native-grid SVM 和 RBF SVM是合理的，但它们是新的对照，不能替换历史名称后当作已完成结果。

### 3.2 当前已保存 Encoder 是交叉熵训练

fewshot_benchmark.py 当前使用 sparse_categorical_crossentropy。v2 第 6.3 节将起点 E0 写作“现有 Encoder + episodic Proto loss”，不符合当前三 seed 的 checkpoint。项目另有 episodic 历史脚本，但需要修复并统一评估后才能作为另一个起点。

建议实验 ID 用 CE-legacy、CE-matched、Episodic、SupCon 等明确名称，避免复用 E0/E1 时混淆。

### 3.3 透过率上限规则会影响几乎整个本地数据集

直接读取当前去重 CSV 得到：

| 数据 | 数值最小/最大 | 至少一个值大于 100 的光谱 | 含非正值的光谱 |
| --- | --- | ---: | ---: |
| C4 | 8.57185 / 121.708 | 2453 / 2649 | 0 |
| C8 | 10.7619 / 143.586 | 2186 / 2390 | 0 |

这些是原始强度值的本地统计，本身不证明信号单位。若已确认它们是 %T，超 100% 要记录和调查；不能仅按计划中的理想 0<T<=1 区间默认整谱删除，否则会删除九成以上数据。也不能任意截成 100、取绝对值、或者将 row_max-y 称作物理吸光度转换。

应分别定义：原单位确认、异常标记、不可转换值、可保留但需解释的测量值、开发阶段固定的处理策略。历史逐谱反转作为 legacy 路线保留；物理转换方案作为单独的版本，不能覆盖旧数据。

### 3.4 共享协方差与 ridge 的定义需一致

v2 写 Cov(Z_base,train)，这包含类间均值差异；当前代码的 prior 是 pooled within-class residual covariance。二者都是可计算量，但代表不同假设，不能用同一个名字混报。

若目标是稳定的类内形状先验，建议明确为：

\[
\Sigma_{base}=N^{-1}\sum_c\sum_{i\in c}(z_i-\mu_c)(z_i-\mu_c)^\top.
\]

随后只在最终混合矩阵加一次 ridge：

\[
\widetilde\Sigma_c=(1-\alpha_c)\Sigma_{base}+\alpha_c\widehat\Sigma_c+\lambda I.
\]

不要在 Sigma_base 已加 epsilon 后又无说明地加入 lambda，造成随 alpha 改变的额外正则。K 或 n_c 应明确指当前保留 exemplar 数，还是所有历史样本数；分母 K/K-1 也要统一。

目前 ftir_research_core.py 已实现 alpha=(n_c-1)/(n_c-1+tau) 的候选收缩公式；v2 第 574 行“目前尚非已有代码公式”需要按当前代码状态更新。它仍未经正式性能验证。

### 3.5 方法选择不能提前看到最终 novel 类

这不只约束 Encoder。若 harmonizer/grid/QC/SVM 参数在六类的带标签开发数据上筛选，再称其中两类从未用于模型选择，也不成立。

需预先规定外层 base/novel 划分和内层开发任务。每个外层任务中，输入方法、损失、tau 等只能由允许的 base 数据或独立开发来源选择；需要“开发时未见类”信号时，可在 base 类内部构造 pseudo-novel 任务。最终两个 novel 类不参与这些选择。

只有六类时，既要足够 base 训练类又要独立开发 novel 与最终 novel，统计能力有限。类别轮换不自动解决这一问题；不能看完所有轮换的 test 再回选一个全局最好模型，却仍将这些轮换称为全新确认性测试。

### 3.6 工程兼容与物理边界

- 当前环境 Python 3.8.20，计划示例中的 str | None 应改为 Optional[str] 等兼容写法，或明确另建 Python>=3.10 环境；无需仅为这个注解升级整个 TensorFlow 环境。
- 现有 cubic 是 interp1d(kind='cubic')，不应未经等价性检查替换为 CubicSpline(natural)。保留 legacy 实现，并将新边界配置独立命名。SciPy CubicSpline 的默认边界确实是 not-a-knot，区别于 natural。[SciPy 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html)
- PCHIP 保形是插值性质，不等于保证物理峰完全正确或分类最好。[SciPy 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
- 抗混叠多相重采样以原采样间隔和 up/down 比例定义输出，不能把非均匀波数数组当作等间隔序列直接处理；前置等间隔化、滤波及边界 padding 都要入配置。[SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
- 若用区间积分/均值，需定义 bin 边缘、输出量是面积还是平均强度，并与点采样插值的差别一起解释。
- 遮点重建应在遮点后才拟合依赖观测值的转换；否则基线/归一化可能提前使用 held-out 强度。若重建的是预处理后的参考谱而非原测量，应明确标注。

## 4. 元数据是主要不可由代码补足的依赖

Xpectrass 论文的原始出版来源确实报告 C4/C8 为同样品的不同分辨率测量，并报告其分析版本 C8 有 607 条相同强度记录。本地曾测得 610 条重复，不应强行改为文献数字。[原始出版来源](https://doi.org/10.1039/D6DD00137H)

它使样品分组成为必要的研究问题，却没有提供本地每条谱的物理样品映射。Sample、文件名、导出行号、scan 和 specimen 必须区分。代码不能凭空制造可靠的 specimen_id；近重复相关性只能产生审查候选，不能自动证明相同实物。

若最终只有很少独立实物，严格 train/test 分组后可能无法维持每类 K+Q。应如实降低实验范围、增加独立数据或限制结论，而不是为了完成矩阵强行拆组。跨分辨率严格结果应写成“元数据允许时完成”，在未能确认分组时不能作为无条件验收项。

## 5. 论文主线应收敛

Xpectrass 已覆盖多来源 FTIR 输入统一、预处理评价和多种 ML 分类。因此“动态 CSV + 公共网格 + SVM 对照”可以是本项目重要工程和方法学基础，但不宜仅凭这些就声称新的核心算法贡献。[Xpectrass](https://doi.org/10.1039/D6DD00137H)

更有区分度的主问题应保留为：经过验证且版本固定的输入接口，能否支持对未见聚合物加类时无需更新 Encoder，并量化精度、旧类干扰和成本。新颖性还需后续更完整的 FSCIL/光谱迁移文献比较，当前审查不构成已证明新颖性的结论。

## 6. 推荐落地顺序和进入条件

1. **继续历史诊断**：现有 diagnose 入口沿用旧输入和旧 episodes，结果明确标历史探索性。不启动新的正式 Encoder 搜索。
2. **WP0 补齐最小输入契约**：schema、真实轴、信号语义、采样间隔、来源/样品身份状态、QC；保存历史数据和处理记录。
3. **WP1 小矩阵**：3 个点插值方法×2 个物理网格；另在确实降采样的粗网格增加一个抗混叠/区间均值方案和 native 对照。先在允许的开发类上跑 SVM 与保真指标。
4. **锁输入配置**：记录 signal conversion、baseline/normalization 顺序、网格、边界条件、异常策略和 content hash。多项一起改变时只能声称整条 pipeline 比较，不把收益单独归因于插值。
5. **少量匹配训练**：同一架构/CE，比较 legacy cubic 与选定配置；之后才在选定配置上比较 episodic/SupCon。输入×架构×损失×网格不一次全展开。
6. **持久 Feature Bank**：在现有 Heads 外增加局部 add/update、save/load、元数据匹配，验证加载前后预测一致，旧类状态不被意外重建，Encoder 权重与 buffer 在内存中逐 Session 保持不变。
7. **正式 FSCIL**：固定类别组合/顺序与同一 query pool，导出 AIA、forgetting、old/new confusion、完整 CM；成本分别测特征提取、实际局部更新和推理。
8. **扩展域验证**：在分组可核验时做 C4↔C8；外部数据通过标签/信号/范围审核后单独做跨来源；变长 Encoder 和 OOD 留为扩展。

当前最应开发的下一块是“可审计的输入契约 + 多方法 harmonizer + 方法/网格开发集比较”，而不是再新增一轮不匹配输入协议的网络训练。
