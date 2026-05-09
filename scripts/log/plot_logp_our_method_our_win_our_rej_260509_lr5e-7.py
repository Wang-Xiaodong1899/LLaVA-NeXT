import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("lr5e-7-ours-win-ours-rej-0514.csv")

# 平滑处理配置
smoothing_window = 20  # 滑动窗口大小（根据数据量调整，建议取总步数的5-10%）

# 应用移动平均（保留原始数据列）
# df['smoothed_chosen'] = df['train/logps/chosen'].rolling(
#     window=smoothing_window, 
#     min_periods=1,        # 允许最小1个数据点开始计算
#     center=True           # 中心对齐模式（使曲线相位不偏移）
# ).mean()
df['smoothed_chosen'] = df['train/logps/chosen']

# df['smoothed_rejected'] = df['train/logps/rejected'].rolling(
#     window=smoothing_window,
#     min_periods=1,
#     center=True
# ).mean()
df['smoothed_rejected'] = df['train/logps/rejected']

df_hound = pd.read_csv("lr0-ours-win-ours-rej-0514.csv")

# 平滑处理配置
smoothing_window = 20  # 滑动窗口大小（根据数据量调整，建议取总步数的5-10%）

# 应用移动平均（保留原始数据列）
# df_hound['smoothed_chosen'] = df_hound['train/logps/chosen'].rolling(
#     window=smoothing_window, 
#     min_periods=1,        # 允许最小1个数据点开始计算
#     center=True           # 中心对齐模式（使曲线相位不偏移）
# ).mean()
df_hound['smoothed_chosen'] = df_hound['train/logps/chosen']

# df_hound['smoothed_rejected'] = df_hound['train/logps/rejected'].rolling(
#     window=smoothing_window,
#     min_periods=1,
#     center=True
# ).mean()
df_hound['smoothed_rejected'] = df_hound['train/logps/rejected']

# df_hound['smoothed_answer'] = df_hound['train/logps/answer'].rolling(
#     window=smoothing_window,
#     min_periods=1,
#     center=True
# ).mean()

# df_input = pd.read_csv("lr5e-7-input-chosen-text-hallu-rej-0514.csv")
# df_input['smoothed_chosen'] = df_input['train/logps/chosen'].rolling(
#     window=smoothing_window, 
#     min_periods=1,        # 允许最小1个数据点开始计算
#     center=True           # 中心对齐模式（使曲线相位不偏移）
# ).mean()

# df_input['smoothed_rejected'] = df_input['train/logps/rejected'].rolling(
#     window=smoothing_window,
#     min_periods=1,
#     center=True
# ).mean()


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
         color='#2b83ba', linewidth=2, label='Win (train)')
plt.plot(df["_step"][:200], df["smoothed_rejected"][:200], 
         color='#d7191c', linewidth=2, label='Lose (train)', linestyle='--')

# plt.plot(df_hound["_step"][:200], df_hound["smoothed_chosen"][:200], 
#          color='#fdae61', linewidth=2, label='Win Hound')
# plt.plot(df_hound["_step"][:200], df_hound["smoothed_rejected"][:200], 
#          color='#abdda4', linewidth=2, label='Lose Hound', linestyle='--')

# plt.plot(df_hound["_step"][:200], df_hound["smoothed_answer"][:200], 
#          color='#000000', linewidth=2, label='GT Hound', linestyle=':')

# plt.plot(df_input["_step"][:200], df_input["smoothed_chosen"][:200], 
#          color='#7570b3', linewidth=2, label='Win Inference')
# plt.plot(df_input["_step"][:200], df_input["smoothed_rejected"][:200], 
#          color='#e7298a', linewidth=2, label='Lose Text hallu', linestyle='--')


# 标注起始点和结束点
# Chosen曲线
# plt.scatter(df["_step"].iloc[0], df["smoothed_chosen"].iloc[0], 
#             color='#2C5F78', marker='*', s=200, edgecolor='black', zorder=5)
# plt.scatter(df["_step"].iloc[199], df["smoothed_chosen"].iloc[199], 
#             color='#2C5F78', marker='*', s=200, edgecolor='black', zorder=5)

# Rejected曲线
# plt.scatter(df["_step"].iloc[0], df["smoothed_rejected"].iloc[0], 
#             color='#E1463C', marker='*', s=200, edgecolor='black', zorder=5)
# plt.scatter(df["_step"].iloc[199], df["smoothed_rejected"].iloc[199], 
#             color='#E1463C', marker='*', s=200, edgecolor='black', zorder=5)

# 标注信息
plt.xlabel('Sample Index', fontweight='bold')
plt.ylabel(r'Implicit reward $\frac{1}{|y|}log\pi_{\theta}(y|x)$', fontweight='bold')
# plt.title(f'Smoothed Log Probabilities (Window Size={smoothing_window})', 
#          fontsize=14, fontweight='bold')
plt.legend()
# plt.ylim(-60, -30)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存输出
plt.savefig('lr0-vs-lr5e-7-ours-win-ours-rej-260509.png')
# plt.show()