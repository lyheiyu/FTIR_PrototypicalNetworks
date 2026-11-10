# FTIR 论文图表包

先打开 figure_catalog.html 浏览所有图，再按英文图注和插入建议选图。原始 FTIR_Manuscript.docx 未修改。

- 01_main_figures：12张候选正文图；png/pdf/svg分格式整理。
- 02_supplementary_figures：网络结构、A–F数据访问、分组角色、特征几何、描述性成本和数据库文件结构，共6张。
- 03_confusion_matrices：三种留出任务×5/10-shot×六种方法，共36页；每页包含A–F六张CM。所有1/5/10/20-shot的CM计数/百分比另有CSV。
- 04_source_tables：每张统计图对应的task/seed或更细级汇总、去重与数据库计数、混淆计数和稿件数值核查。
- 05_manuscript_notes：中文稿件检查、数据库结构、英文图注、原稿纯文本快照。
- source：可复现制图脚本与论文图表manifest。

图仅由现有数据与保存结果生成，没有重训模型、重拟合PCA或新增分类实验。图中误差棒为相关task/seed均值的描述性SD，不是独立样本置信区间。CM是重复预测的池化计数；每个真类9,000次查询出现不等于9,000个独立实物。PCA128与M1均128维，但尺度和相对正则强度不同。

复现（在项目根目录）：

```powershell
& 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe' -X utf8 build_ftir_manuscript_figures_v1.py --output 'paper draft/FTIR_Manuscript_Figures_v1'
```

原始数据、旧实验结果和原稿均被只读访问；输入哈希记入figure_manifest.json。全部图可用于排版；具体主文/补充取舍见稿件检查。
