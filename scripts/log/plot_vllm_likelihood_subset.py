import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("llava-next-dpo-0228.csv")

# 1) 时间顺序
df = df.sort_values("_step").reset_index(drop=True)

chosen_col = "train/logps/chosen"
rejected_col = "train/logps/rejected"

# 2) 找“最早的最接近点”
diff = (df[chosen_col] - df[rejected_col]).abs()
idx0 = diff.idxmin()   # pandas 默认返回最早的 min
picked_step = df.loc[idx0, "_step"]

# 3) 只取该点及其之后的 50 个点
sub = df.iloc[idx0: idx0 + 50].copy().reset_index(drop=True)

# 4) 只对这 50 个点做平滑
smoothing_window = 30  # 50 个点下，10 很合适
sub["chosen_s"] = sub[chosen_col].rolling(
    window=smoothing_window, min_periods=1, center=True
).mean()
sub["rejected_s"] = sub[rejected_col].rolling(
    window=smoothing_window, min_periods=1, center=True
).mean()

# 5) 起点对齐（两个曲线在 t=0 重合）
sub["chosen_align"] = sub["chosen_s"] - sub["chosen_s"].iloc[0]
sub["rejected_align"] = sub["rejected_s"] - sub["rejected_s"].iloc[0]

# ===== 绘图 =====
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
    'figure.figsize': (10, 5),
    'figure.dpi': 300,
})

plt.figure()

plt.plot(sub["_step"], sub["chosen_align"],
         linewidth=2, label="Winning")
plt.plot(sub["_step"], sub["rejected_align"],
         linewidth=2, linestyle="--", label="Losing")

# 起点（重合）
plt.scatter(sub["_step"].iloc[0], 0,
            marker='*', s=220, edgecolor='black', zorder=5)

plt.xlabel("Training Step", fontweight="bold")
plt.ylabel(r"Aligned Likelihood $\Delta \log\pi_{\theta}(y|x)$", fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("closest_point_50_aligned_smoothed.png")
