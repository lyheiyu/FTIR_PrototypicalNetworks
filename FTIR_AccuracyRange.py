import matplotlib.pyplot as plt
import numpy as np

# 示例数据
models = ['Model A', 'Model B', 'Model C']
accuracy = [0.85, 0.78, 0.92]
error = [0.05, 0.04, 0.03]  # 误差范围

# 转换为 NumPy 数组以便于处理
models_np = np.arange(len(models))

# 创建图形
fig, ax = plt.subplots()

# 绘制带误差条的折线图
ax.errorbar(models_np, accuracy, yerr=error, fmt='-o', capsize=5, color='skyblue', ecolor='black', elinewidth=2, markeredgewidth=2)

# 添加标题和标签
ax.set_title('Model Accuracy with Error Bars')
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')

# 设置 x 轴刻度标签
ax.set_xticks(models_np)
ax.set_xticklabels(models)

# 设置 y 轴范围为 0 到 1
ax.set_ylim([0, 1])

# 显示图形
plt.show()