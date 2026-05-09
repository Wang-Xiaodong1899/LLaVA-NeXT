import matplotlib.pyplot as plt
import numpy as np

# 数据
ls_values = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9]
overall_scores = [42.0, 42.0, 43.6, 43.2, 43.0, 42.2, 42.4]

# 创建图形
plt.figure(figsize=(10, 4))

# 绘制折线图
plt.plot(ls_values, overall_scores, marker='o', linewidth=2, markersize=8, color='steelblue')

# 标记最佳性能点
best_idx = overall_scores.index(max(overall_scores))
plt.scatter(ls_values[best_idx], overall_scores[best_idx], 
           s=150, color='red', zorder=5, label=f'Best: {overall_scores[best_idx]}')

# 添加数据标签
for i, (x, y) in enumerate(zip(ls_values, overall_scores)):
    plt.annotate(f'{y}', (x, y), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=10)

# 设置标题和标签
plt.title('Overall Performance vs Label Smoothing (α) on Video-MME', fontsize=14, fontweight='bold')
plt.xlabel('Label Smoothing Coefficient (α)', fontsize=12)
plt.ylabel('Overall Score', fontsize=12)

# 设置坐标轴
plt.xlim(-0.05, 0.95)
plt.ylim(41.5, 44.0)
plt.grid(True, alpha=0.3, linestyle='--')


# 添加图例
plt.legend(fontsize=11)

# 显示图形
plt.tight_layout()
# plt.show()

# 可选：保存图像
plt.savefig('label_smoothing_performance.png', dpi=300, bbox_inches='tight')