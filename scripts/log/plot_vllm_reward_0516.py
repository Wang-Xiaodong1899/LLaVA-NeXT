import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
# df = pd.read_csv("llava-next-dpo.csv")
df = pd.read_csv("llava-next-dpo-0228.csv")

# 平滑处理配置
smoothing_window = 20  # 滑动窗口大小（根据数据量调整，建议取总步数的5-10%）

# 应用移动平均（保留原始数据列）
df['smoothed_chosen'] = df['train/rewards/chosen'].rolling(
    window=smoothing_window, 
    min_periods=1,        # 允许最小1个数据点开始计算
    center=True           # 中心对齐模式（使曲线相位不偏移）
).mean()

df['smoothed_rejected'] = df['train/rewards/rejected'].rolling(
    window=smoothing_window,
    min_periods=1,
    center=True
).mean()

# 设置科研绘图风格
# plt.style.use('seaborn-v0_8')
plt.style.use('default')
sns.set_palette("tab10")
plt.rcParams.update({
    'font.family': 'Times New Roman', 
    'font.size': 20,          # 基础字体大小
    'axes.titlesize': 20,     # 标题字号
    'axes.labelsize': 20,     # 坐标轴标签字号
    'xtick.labelsize': 12,    # X轴刻度字号
    'ytick.labelsize': 12,    # Y轴刻度字号
    'legend.fontsize': 20,    # 图例字号
    'figure.figsize': (10, 5), # 调整画布尺寸以适应大字体
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.facecolor': 'white',          # 图像背景为白色
    'figure.facecolor': 'white',        # 画布背景为白色
    'grid.color': 'gray',               # 网格线颜色
    'grid.linestyle': '--',             # 网格线样式
    'grid.alpha': 0.3                   # 网格线透明度
})


# 创建画布
plt.figure()

# 绘制平滑曲线
plt.plot(df["_step"][:200], df["smoothed_chosen"][:200], 
         color='#2C5F78', linewidth=2, label='Winning')
plt.plot(df["_step"][:200], df["smoothed_rejected"][:200], 
         color='#E1463C', linewidth=2, label='Losing', linestyle='--')
plt.plot(df["_step"][:200], df["smoothed_chosen"][:200]-df["smoothed_rejected"][:200], 
         color='#D8A215', linewidth=2, label=r'Margin', linestyle='--')


# 标注起始点和结束点
# Chosen曲线
plt.scatter(df["_step"].iloc[0], df["smoothed_chosen"].iloc[0]-df["smoothed_rejected"].iloc[0], 
            color='#D8A215', marker='*', s=200, edgecolor='black', zorder=5)

plt.scatter(df["_step"].iloc[0], df["smoothed_chosen"].iloc[0], 
            color='#2C5F78', marker='*', s=200, edgecolor='black', zorder=5)
plt.scatter(df["_step"].iloc[199], df["smoothed_chosen"].iloc[199], 
            color='#2C5F78', marker='*', s=200, edgecolor='black', zorder=5)

# Rejected曲线
plt.scatter(df["_step"].iloc[199], df["smoothed_chosen"].iloc[199]-df["smoothed_rejected"].iloc[199], 
            color='#D8A215', marker='*', s=200, edgecolor='black', zorder=5)

plt.scatter(df["_step"].iloc[0], df["smoothed_rejected"].iloc[0], 
            color='#E1463C', marker='*', s=200, edgecolor='black', zorder=5)
plt.scatter(df["_step"].iloc[199], df["smoothed_rejected"].iloc[199], 
            color='#E1463C', marker='*', s=200, edgecolor='black', zorder=5)

# 标注信息
plt.xlabel('Training Step', fontweight='bold')
# plt.ylabel('Implicit reward', fontweight='bold')
plt.ylabel(r'Implicit reward $log\frac{\pi_{\theta}(y|x)}{\pi_{sft}(y|x)}$', fontweight='bold')
# plt.title(f'Smoothed Log Probabilities (Window Size={smoothing_window})', 
#          fontsize=14, fontweight='bold')
plt.legend()
# plt.ylim(-60, -30)
plt.ylim(-2.25, 0.45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存输出
plt.savefig('vllm_200_reward_add-0516.png')
# plt.show()