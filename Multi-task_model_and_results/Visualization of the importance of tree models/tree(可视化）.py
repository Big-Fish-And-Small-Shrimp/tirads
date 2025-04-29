import matplotlib.pyplot as plt

# 定义特征标签和对应的重要性
feature_labels = ['COM',
    'ECH',
    'SHA',
    'MAR',
    'ELCA',
    'MAC',
    'PCA',
    'PEF']
# feature_importances = [0.53, 0.82, 0.36, 0.27, 0.49, 0.19, 0.16, 0.10]
# feature_importances = [0.37, 0.42, 0.69, 0.47, 0.13, 0.04, 0.07, 0.41]
feature_importances = [0.35, 0.49, 0.56, 0.35, 0.10, 0.09, 0.08, 0.35]
# 由于我们要将特征标签放在y轴上，因此需要将数据“转置”
# 这里我们其实不需要真正地进行数学上的转置，而是改变绘图时的坐标轴

# 创建条形图，注意这里是水平条形图（horizontal bar）
fig, ax = plt.subplots()
# bars = ax.barh(feature_labels, feature_importances, color='skyblue', edgecolor='black',height=0.3)
# bars = ax.barh(feature_labels, feature_importances, color='orangered', edgecolor='black',height=0.3)
bars = ax.barh(feature_labels, feature_importances, color='gold', edgecolor='black',height=0.3)
# 在条形旁边添加数据标签
for bar, importance in zip(bars, feature_importances):
    xval = bar.get_width()  # 对于水平条形图，宽度对应x轴的值
    yval = bar.get_y() + bar.get_height() / 2 - 0.05  # 计算标签的y轴位置，稍微向下偏移以避免重叠
    ax.annotate(f'{importance:.2f}', xy=(xval, yval),
                xytext=(5, 0),  # 5 points horizontal offset
                textcoords="offset points",
                ha='left', va='center')  # 标签在条形图的右侧

# 设置y轴标签和标题
ax.set_xlim(0, 1)
ax.set_yticks(range(len(feature_labels)))
ax.set_yticklabels(feature_labels)
ax.invert_yaxis()  # 反转y轴，使得第一个标签在顶部
ax.set_xlabel('Importance')
ax.set_title('Feature Importance')

# 添加网格线（可选）
ax.grid(axis='x', linestyle='--', alpha=0.7)

# 显示图表并调整布局
plt.tight_layout()
plt.savefig("tree importance")