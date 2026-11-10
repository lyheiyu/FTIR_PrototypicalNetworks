# v2 新测试：输入方法与物理网格的开发集筛选

此前 frozen_diagnosis_v1 和 c4_incremental_v1 属于上一版实验。新入口为 run_harmonization_v2.py；此入口不训练新 Encoder，也不重复上一版类别增量实验。

## 当前实现范围

- Native 原生网格参照（仍使用同一谱值预处理）、Linear、legacy interp1d cubic、PCHIP、粗网格 piecewise-linear 区间均值。
- 同一组 base 类、相同 tune/screen split、support/query ID、SVM 参数搜索预算。
- 物理网格由原始波数范围和采样间隔构造，默认网格间隔为原生中位间隔的 2/4 倍。实际范围、点数和步长写入 timing.csv 和 *_grid.npy。它不表示仪器分辨率被改变。
- 只用显式指定的 base 类做开发筛选，其他类别排除，不写最终 test 最优结论。当前也不自动把开发筛选结果应用于旧 Encoder。
- 有显式 schema 的两列/多强度列竖表和首行波数宽表解析 API，在 ftir_harmonization_v2.py 的 parse_ftir_csv；当前实验 CLI 仍使用已审计本地 registry 数据集，尚不是任意目录批量导入工具。
- 全谱主峰位移/峰高、面积、越过原谱全局强度范围的过冲、原生观测点遮点重建、处理耗时和矩阵大小。这里尚不是 v2 完整的化学关键峰匹配或局部缺失区间 QC。
- bin_mean 是粗网格区间均值，不是理想带限滤波；输出语义与点插值不同，配置中明确记录，面积按 bin 宽加权。它不参加遮点的点值插值 RMSE 排名。

## 环境

在项目根目录，激活 fewshotlearning 后执行下列 python 命令；也可将 python 替换为：

```powershell
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe'
```

## 先小跑

```powershell
python run_harmonization_v2.py --dataset c4 --output results/harmonization_v2_smoke --base-classes PET PVC PP PS --signal legacy --seeds 0 --shots 5 --episodes 3 --val-episodes 2
```

只用于检查流程，不用作论文结论。输出目录必须不存在。本次已完成相同规模的程序检查，保存在 results/harmonization_v2_code_check，不必覆盖它。

## 再运行开发集比较

```powershell
python run_harmonization_v2.py --dataset c4 --output results/harmonization_v2_dev_c4 --base-classes PET PVC PP PS --signal legacy --seeds 0 1 2 --shots 1 5 10 20 --episodes 100 --val-episodes 5
```

这里明确保留 HDPE/LDPE 不参与输入方法选择。该配置仅适用于“这四类是允许开发的 base 类”的外层任务。若要轮换其他 base/novel 类，必须为相应任务重新限定允许的开发类，或者先在独立开发来源上锁定一个全局配置；不能在六类都参加过筛选后声称所有轮换的新类从未用于模型选择。

如下一步选择在 C8 的对应 base 类上重复开发比较：

```powershell
python run_harmonization_v2.py --dataset c8 --output results/harmonization_v2_dev_c8 --base-classes PET PVC PP PS --signal legacy --seeds 0 1 2 --shots 1 5 10 20 --episodes 100 --val-episodes 5
```

这属于各自来源的开发实验，不是已经完成 C4→C8 的冻结迁移。C8 的派生网格与 C4 未必完全相同，跨来源共享网格必须另行锁定，不可直接把这两份结果称为同一网格的跨域比较。

## 信号和分组边界

`--signal` 必须显式指定：legacy、absorbance、transmittance_percent、transmittance_fraction。

当前建议先用 legacy，含义是保留原 profile 的逐谱反转、native airPLS 和 maxabs，再只改变 harmonizer。它不声称完成物理吸光度转换。只有原始单位确认后，才使用物理转换模式；不能仅因数值大小猜单位。

%T/fraction T 转吸光度时不允许非正值。T>100% 默认拒绝并记录 signal_qc.csv；若经过确认要保留，必须明确 `--transmittance-upper flag`，保留负吸光度并打标记，不默默截断为 100%。物理转换和 legacy 对比属于完整 pipeline 变化，不是仅插值算法变化。

如果有核实过的 specimen 映射，可以提供 `--groups verified_groups.csv`，格式为 dataset,row_id,group。每个 base 类的同一物理组只保留一个代表，防止重复测量跨开发 split；未提供时只做文件内数值去重，不声称严格实物独立性。

## 输出怎么看

- summary.csv：各方法×网格×shot 的 Accuracy/Macro-F1 均值和 seed 标准差。
- method_grid_development.png/pdf：开发分类曲线。
- *_CM.png、confusions.npz、confusion_index.json：开发集逐类混淆矩阵，默认画 5-shot。
- spectral_fidelity.csv：原生点遮点重建、主峰/面积/全局范围过冲诊断。
- timing.csv：实际采样步长、点数、harmonization 耗时和输出矩阵字节数。
- sample_index.csv、development_splits.csv、episodes.json、逐候选预测 CSV：用于逐条追溯和配对比较。
- svm_tuning.csv、svm_selection.csv：每个 C 的 tune 分数及选定参数。SVM 在每个 episode 仅拟合 support。
- DEVELOPMENT_SHORTLIST.json：按开发集 Macro-F1 的候选排序；不是 final-test 冠军或可直接部署的输入配置。

遮点重建发生在原始强度数组上，在任何逐谱基线/归一化之前留点，避免隐藏点参与拟合。它与采用 legacy 预处理谱的分类/主峰诊断具有不同量纲和目的，不能混成一个综合物理保真结论。Native 的 RMSE=0 只是参照标记，不参与“重建方法最好”的判断。

开发结果筛选后，仍需结合 QC、关键峰、实物身份和信号单位人工审核，再锁定版本化输入配置；之后才公平地重训匹配 Encoder。当前未实现完整带限滤波、SupCon、持久 Feature Bank 或完整 v2 部署接口。

## 测试

```powershell
python -m unittest discover -s tests -p "test_harmonization_v2.py" -v
```

覆盖显式 CSV 格式/单位、升降序、重复轴、信号转换和异常策略、legacy cubic 等价性、禁止外推、区间面积、遮点重建、排除未来类和配对预测导出。
