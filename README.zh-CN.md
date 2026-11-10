# FTIR Prototypical Networks

[English](README.md) | **简体中文**

用于 FTIR 聚合物少样本分类、冻结特征表示与新类别增量注册的研究代码。项目比较光谱预处理、编码器结构、欧氏与马氏原型分类头，以及 C4/C8 数据之间的迁移表现。

> 建议先阅读下方最终实验报告。仓库保留了历史代码和开发实验，它们使用不同协议，不能直接合并为同一个基准结果。

## 从哪里看结果

| 想了解的问题 | 报告入口 |
| --- | --- |
| Episodic ProtoNet 与监督交叉熵（CE）训练有什么区别？ | [最终 ProtoNet 中文总结](results/protonet_bidirectional_final/summary_zh.md) · [英文论文结果](results/protonet_bidirectional_final/manuscript_results_en.md) |
| C4/C8 双向迁移效果如何？ | [最终双向迁移结果](results/bidirectional_c4_c8_final/final_results_zh.md) |
| 收缩马氏距离是否优于完整类协方差？ | [完整协方差消融结论](results/mahalanobis_full_ablation_v1/conclusion.md) |
| Linear、Cubic、PCHIP 插值方法如何比较？ | [插值方法与网格比较结论](results/harmonization_v2_dev_c4/RESULTS_CONCLUSIONS.md) |
| 不同编码器结构的效果如何？ | [编码器结构比较](results/encoder_architecture_v1/architecture_comparison.md) |
| 冻结编码器后能否直接注册新类？ | [原型分类器研究](results/external_transfer_prototype_v1/results_overview_zh.md) · [增量分类研究](results/frozen_incremental_v1/results_overview_zh.md) |
| 面对异构光谱输入时表现如何？ | [异构 FTIR 迁移结果](results/heterogeneous_ftir_transfer_v1/results_overview_zh.md) |

插值开发实验中，同一网格上的分类差异很小；Cubic 的遮点重建误差较低，Linear 的插值开销较小。最终 ProtoNet 实验中，收缩与完整协方差存在基类/新类之间的取舍：新类召回更好，不一定意味着整体增量准确率更高。具体数值与适用范围见对应报告。

## 方法流程

```text
FTIR 光谱 → 预处理 / 统一物理网格 → 特征表示
                                     ↓
                  支持样本 → 每类原型 + 距离矩阵
                                     ↓
                       查询样本 → 距离最小的类别
```

特征表示包括预处理光谱、PCA 和神经编码器。在冻结增量评估中，新类别通过少量支持样本注册，不更新编码器，也不重新拟合已注册的原型类别。

收缩马氏分类头为类别 c 保存支持集均值 μc，以及完整精度矩阵：

```text
Sc     = Σi (zi − μc)(zi − μc)ᵀ / nc
αc     = (nc − 1) / (nc − 1 + τ)
Σ̃c     = αc Sc + (1 − αc) Σbase + λI
d²c(z) = (z − μc)ᵀ Σ̃c⁻¹ (z − μc)
```

最终对照使用 τ = 5、λ = 0.001。Σbase 仅由源域基类训练特征的类内残差汇总估计。样本少时更依赖基类先验，样本多时更依赖该类自身协方差。完整马氏使用 Sc + λI；共享马氏让所有类别使用同一个 Σbase + λI。最终 episodic 编码器在训练阶段使用欧氏原型距离，马氏分类头在训练完成后应用。

## 文件导航

### 公共实现

| 文件 / 目录 | 用途 |
| --- | --- |
| [ftir_research_data.py](ftir_research_data.py) | 数据集注册、标签、光谱读取和预处理工具。 |
| [ftir_research_core.py](ftir_research_core.py) | 协方差先验、参考分类器、分类头调参与评估。 |
| [frozen_prototype_heads_v1.py](frozen_prototype_heads_v1.py) | 按类别注册的欧氏、余弦、共享和收缩原型分类头。 |
| [run_full_mahalanobis_ablation_v1.py](run_full_mahalanobis_ablation_v1.py) | 完整类协方差分类头，以及配对消融实验入口。 |
| [encoder_architecture_models.py](encoder_architecture_models.py) | M0/M1 编码器与光谱分箱池化。 |
| [protonet_training_final.py](protonet_training_final.py) | 最终 ProtoNet 对照中的 episodic 编码器训练。 |
| [bidirectional_protocol_final.py](bidirectional_protocol_final.py) | C4/C8 配对实验共用协议。 |
| [ftir_harmonization_v2.py](ftir_harmonization_v2.py) | 光谱解析、波数轴清理、信号转换和重采样。 |
| [heterogeneous_ftir_adapter_v1.py](heterogeneous_ftir_adapter_v1.py) | 不同光谱覆盖范围的输入适配。 |
| [deduplicate_plastics500.py](deduplicate_plastics500.py) | Plastics500 输入数据去重。 |
| [PLS.py](PLS.py)、[utils.py](utils.py) | 基线校正、公共指标与绘图工具。 |
| [research_configs/](research_configs/) | 研究任务配置。 |
| [tests/](tests/) | 数据处理、距离分类头和实验协议检查。 |

### 实验运行入口

| 脚本 | 实验内容 / 操作说明 |
| --- | --- |
| `fewshot_benchmark.py` | 早期数据集内部配对少样本基准。[说明](FEWSHOT_RUN.md) |
| `run_ftir_research.py` | 表征诊断与早期增量评估。[说明](FTIR_RESEARCH_RUN.md) |
| `run_harmonization_v2.py` | 插值方法与目标网格开发比较。[说明](HARMONIZATION_V2_RUN.md) |
| `run_encoder_architecture_v1.py` | 匹配条件的编码器结构比较。[说明](ENCODER_ARCHITECTURE_V1_RUN.md) |
| `run_frozen_incremental_v1.py` | 冻结特征增量分类和更新成本。[说明](FROZEN_INCREMENTAL_V1_RUN.md) |
| `run_external_prototype_v1.py` | 原型注册与参考基线。[说明](EXTERNAL_PROTOTYPE_V1_RUN.md) |
| `run_full_mahalanobis_ablation_v1.py` | 完整类协方差与共享/收缩分类头比较。[说明](FULL_MAHALANOBIS_ABLATION_V1_RUN.md) |
| `run_heterogeneous_ftir_transfer_v1.py` | 异构数据迁移。[说明](HETEROGENEOUS_FTIR_TRANSFER_V1_RUN.md) |
| `run_bidirectional_c4_c8_final.py` | 最终 C4/C8 配对监督编码器实验。[说明](BIDIRECTIONAL_C4_C8_FINAL_RUN.md) |
| `run_protonet_bidirectional_final.py` | 最终 episodic ProtoNet 与 CE 参考对照。[说明](PROTONET_BIDIRECTIONAL_FINAL_RUN.md) |
| `run_pca128_shrinkage_bidirectional_v1.py` | PCA128 + 收缩马氏的补充比较。 |
| `supplement_heterogeneous_full_mahalanobis_v1.py` | 异构迁移实验中的完整协方差补充对照。 |

`analyze_*.py` 汇总对应实验的已保存产物；`audit_*.py` 核对协议和来源记录。`plot_fewshot_benchmark.py` 绘制基准图；`build_ftir_manuscript_figures_v1.py` 整理论文图。这些脚本可能需要未上传 GitHub 的本地产物。

### 论文与历史代码

[paper draft/](paper%20draft/) 包含论文草稿、研究计划和图件集合。这些是工作文档，不代表论文已发表。[GITHUB_SYNC.md](GITHUB_SYNC.md) 说明同步范围。

根目录保留了早期 `FTIR_*.py` 实验，涉及 ProtoNet、马氏距离、MAML、孪生网络和数据增强，包括 `FTIR_AugmentationBasedOnEMSA.py` 与 `FTIR_AugmentationBasedOnPCA_EMSA.py`。其他独立示例、应用与演示文件属于历史工具。部分旧脚本导入时会直接运行，或包含本机路径；当前实验优先使用上表中有说明文档的入口。

## 环境与首次检查

使用与 TensorFlow 2.13 兼容的 Python 环境。研究入口主要依赖 TensorFlow/Keras、NumPy、pandas、SciPy、scikit-learn、Matplotlib 和 threadpoolctl。`requirements.txt` 是历史混合项目依赖列表，不是最终实验的最小环境锁定文件，其中还包含应用、CAD 与 PyTorch 等其他用途的依赖。

在项目根目录检查已有环境：

```powershell
python -c "import tensorflow as tf; import numpy, pandas, scipy, sklearn, matplotlib, threadpoolctl; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
python -m unittest discover -s tests -p test_harmonization_v2.py -v
```

以上用于环境和协议检查，不是完整训练。涉及真实数据的运行需要先准备本地数据，并阅读对应实验说明。例如，最终双向实验从小规模流程检查开始：

```powershell
python run_bidirectional_c4_c8_final.py --smoke --output results/bidirectional_c4_c8_review_smoke
```

使用新的输出目录。启动正式运行前阅读说明；后续实验依赖前序保存的模型与实验清单。

## 数据与可复现范围

- 仓库不分发原始数据。`ftir_research_data.py` 的注册表记录预期输入；C4/C8 去重 CSV 位于本地 `dataset/deduplicated/`。
- GitHub 保存代码、运行说明、论文草稿、部分报告、汇总表和图。模型权重、特征数组、详细 episode 产物和 smoke/check 输出保留在本地。
- 克隆仓库可阅读方法和已上传报告，但它**不是完整实验归档**。部分分析脚本需要额外的本地产物。
- 早期数据集内部评估与后期未见类别增量协议回答不同问题。K-shot 指支持集预算，不等于训练编码器时总共只使用 K 条标签。
- 最终结果属于探索性评估。记录/重复组不等于已验证的独立实物；C4/C8 比较也不能证明独立仪器之间的泛化。具体限制见各实验报告。

## 更新到 GitHub

提交前检查文件列表：

```powershell
git status --short
git add --all
git diff --cached --stat
git commit -m "Update research code and reports"
git push origin main
```

大型本地产物由 `.gitignore` 排除。选定的结果与论文 CSV 使用普通 Git 存储，规则见 `.gitattributes`。
