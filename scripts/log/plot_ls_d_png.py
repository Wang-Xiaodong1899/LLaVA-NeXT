import matplotlib.pyplot as plt
import numpy as np

# 实验1数据：Label Smoothing (ls)
ls_values = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9]
ls_overall_scores = [42.0, 42.0, 43.6, 43.2, 43.0, 42.2, 42.0]

# 实验2数据：Dropout (d)
d_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
d_overall_scores = [41.9, 42.0, 42.1, 41.9, 41.8, 41.7]

# 创建图形
plt.figure(figsize=(11, 4))

# 绘制第一个实验的折线图 (Label Smoothing)
plt.plot(ls_values, ls_overall_scores, marker='o', linewidth=2, markersize=8, 
         color='steelblue', label='Label Smoothing (ls)')

# 绘制第二个实验的折线图 (Dropout)
plt.plot(d_values, d_overall_scores, marker='s', linewidth=2, markersize=8,
         color='coral', label='Threshold (d)')

# 标记最佳性能点
ls_best_idx = ls_overall_scores.index(max(ls_overall_scores))
d_best_idx = d_overall_scores.index(max(d_overall_scores))

plt.scatter(ls_values[ls_best_idx], ls_overall_scores[ls_best_idx], 
           s=150, color='darkblue', zorder=5, 
           label=f'Best ls (α={ls_values[ls_best_idx]})')

plt.scatter(d_values[d_best_idx], d_overall_scores[d_best_idx], 
           s=150, color='darkred', zorder=5,
           label=f'Best d (d={d_values[d_best_idx]})')

# 为Label Smoothing添加数据标签
for i, (x, y) in enumerate(zip(ls_values, ls_overall_scores)):
    plt.annotate(f'{y}', (x, y), textcoords="offset points", 
                xytext=(0, 12), ha='center', fontsize=9, color='steelblue')

# 为Dropout添加数据标签
for i, (x, y) in enumerate(zip(d_values, d_overall_scores)):
    plt.annotate(f'{y}', (x, y), textcoords="offset points", 
                xytext=(0, -20), ha='center', fontsize=9, color='coral')

# 设置标题和标签
# plt.title('Overall Performance: Label Smoothing (α) vs Threshold (d) on Video-MME', 
#           fontsize=15, fontweight='bold')
plt.xlabel('Parameter Value', fontsize=13)
plt.ylabel('Overall Score', fontsize=13)

# 设置坐标轴
plt.xlim(-0.02, 1.02)
plt.ylim(41.0, 44.0)
plt.grid(True, alpha=0.3, linestyle='--')

# 添加次要网格线
plt.minorticks_on()
plt.grid(True, which='minor', alpha=0.1, linestyle=':')

# 添加垂直线区分两个实验的x轴范围
plt.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 添加图例
plt.legend(fontsize=11, loc='upper right')

# 添加文本标注说明两个实验
plt.text(0.45, 43.7, 'Exp1: Label Smoothing Experiment', ha='center', fontsize=11, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
plt.text(0.55, 41.2, 'Exp2: Threshold Experiment', ha='center', fontsize=11,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

# 显示图形
plt.tight_layout()
# plt.show()
plt.savefig('label_smoothing_d_performance.png', dpi=300, bbox_inches='tight')

# # 打印详细分析
# print("="*60)
# print("性能对比分析:")
# print("="*60)
# print("\nLabel Smoothing 实验:")
# print("-"*30)
# ls_best_score = max(ls_overall_scores)
# ls_best_ls = ls_values[ls_overall_scores.index(ls_best_score)]
# print(f"最高性能: {ls_best_score} (ls={ls_best_ls})")
# print(f"最低性能: {min(ls_overall_scores)}")
# print(f"性能范围: {ls_best_score-min(ls_overall_scores):.2f}")

# print("\nDropout 实验:")
# print("-"*30)
# d_best_score = max(d_overall_scores)
# d_best_d = d_values[d_overall_scores.index(d_best_score)]
# print(f"最高性能: {d_best_score} (d={d_best_d})")
# print(f"最低性能: {min(d_overall_scores)}")
# print(f"性能范围: {d_best_score-min(d_overall_scores):.2f}")

# print("\n" + "="*60)
# print(f"性能对比:")
# print(f"Label Smoothing最佳 vs Dropout最佳: {ls_best_score} vs {d_best_score}")
# print(f"差值: {ls_best_score - d_best_score:.2f}")
# print("="*60)