# 稿件与图表检查

已读取 FTIR_Manuscript.docx 的全部段落和6张表。目前文件中没有嵌入图片。稿件已经保留 PCA 优于网络、episodic 欧氏新类识别未改善、收缩在 C/D 中存在基类取舍等关键负结果；不需要为了叙事再增加实验。

## 优先补充或修正

1. **数据去重有两个层次。** C4/C8 原始各3,000条；按单库精确数值相等去重后分别2,649/2,390条，移除重复行351/610条。原始数据库未删除。随后跨分辨率共享记录ID和两库重复别名传递合并，形成2,047个保守记录组；每组每个分辨率选一个代表参与最终实验。不能把2,047称为已验证的独立实物数量。Fig01与数据计数CSV可直接支撑这一段。
2. **2.1 的分组描述需更精确。** 目前“training, validation, support and query were separated”容易让读者认为所有支持集都与训练集独立。实际上20条/类的 base reference 来自 base_train；新类的 support/query 池互斥，query 与 train/validation 均隔离。图中已明确这一点。
3. **2.2 漏写第一步变换。** 实际顺序为原生轴排序 → max(y)−y → 原生轴airPLS基线扣除 → 原生轴max-abs归一化 → cubic插值到固定1,868坐标。airPLS默认 lambda=100、porder=1、itermax=15。请避免把未知单位直接标为吸光度；光谱图已采用 preprocessed signal (a.u.)。
4. **2.4 可直接采用附图中的精确结构。** M1为三分支Conv1D(16,k=9/25/49)，两次[MaxPool2 + Conv32,k=5]，Conv8,k=1，32个有序位置bin，Flatten256，Dense128/ReLU；编码器47,400参数。CE临时四分类头额外516参数，评估时丢弃。
5. **同域/迁移与数据访问须分清。** E/F额外需要每基类20条目标分辨率参考，即80条目标基类标签，不是无标签域适应。PCA/M1拟合可访问合法的全部base-train，而原谱1-NN/SVM分类库仅使用每基类20条和每新类K条。分类参考配对不等于所有算法的初始训练数据访问完全相同。
6. **图中的误差棒不是独立样本置信区间。** 它们是9个相关task/seed均值的描述性SD；不能把200次episode当200个独立实物。CM展示池化重复预测百分比，不应把计数解释成新的样本量。
7. **成本措辞。** 0.279/0.277 ms应称 batch64 下按光谱分摊的推断时间，且是每模型中位数再求均值，不是单条输入的独立调用延迟。训练、PCA拟合与局部注册计时范围不同，只作描述性对照。
8. **投稿前仍需核对原始来源。** 吸光度/透过率语义、原始采集仪器与样本独立性、作者信息和参考文献未在此次制图中外部核验。不能仅凭画图补成已验证事实。原稿末尾 author verification notes 需在完成核对后删除。

## 数值检查

表2–5已从源CSV按未四舍五入的task/seed均值复核；详见 manuscript_numeric_rounding_audit.csv。

- 表4，第5个数据行，第2列：90.02 应为 **90.01**。
- 表5，第1个数据行，第1列：99.12 应为 **99.11**。
- 表5，第2个数据行，第4列：−3.42 应为 **-3.41**。

## 可替换的数据段落（英文）

Within each resolution subset, exact numeric duplicate spectra were removed without rounding or a similarity threshold, retaining 2,649 C4 and 2,390 C8 records from 3,000 records each. The original files were preserved. Shared original record identifiers and both duplicate-alias maps were then unioned conservatively across resolutions, yielding 2,047 record components. One representative per component and resolution was used. These components provide record-level blocking, not verified physical specimen identities. Base training, validation and query groups were disjoint; the 20 base reference spectra per class were sampled from the legal base-training groups. Held-out novel classes used separate support and query pools and were excluded from representation fitting and source validation.

## 图表选用建议

正文推荐Fig01、03、05、06、07、09、10，再选Fig11或Fig12的一张核心混淆图；Fig02光谱、Fig04传统基线、Fig08全shot曲线可按期刊篇幅移入补充。FigS01–S06与按任务整理的完整CM作为补充材料。不要把同一结果的大表和全部对应图都重复塞入正文。图号只是本文件夹的稳定编号，正式投稿时可按最终选图重新连续编号。
