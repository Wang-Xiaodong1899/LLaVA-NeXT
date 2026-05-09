import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("llava-next-dpo-0228.csv")

# 1) 按时间排序
df = df.sort_values("_step").reset_index(drop=True)

chosen_col = "train/logps/chosen"
rejected_col = "train/logps/rejected"

# 2) 找“最早的最接近点”
diff = (df[chosen_col] - df[rejected_col]).abs()
idx0 = diff.idxmin()

# 3) 只取该点及之后 50 个点
N = 50
sub = df.iloc[idx0: idx0 + N].copy().reset_index(drop=True)

# 4) 横轴从 0 开始（相对 step）
sub["rel_step"] = sub["_step"] - sub["_step"].iloc[0]

# 5) 只对这 50 个点做平滑
smoothing_window = 30  # 50 点下建议 5~15，自行调
sub["chosen_s"] = sub[chosen_col].rolling(
    window=smoothing_window, min_periods=1, center=True
).mean()
sub["rejected_s"] = sub[rejected_col].rolling(
    window=smoothing_window, min_periods=1, center=True
).mean()

# 6) 让两条曲线“起点重合”：平移下面那条（这里平移 Losing）
shift = sub["chosen_s"].iloc[0] - sub["rejected_s"].iloc[0]
sub["rejected_s_shifted"] = sub["rejected_s"] + shift

# ===== 绘图（纵轴仍是 log-likelihood 的正常值，只是 Losing 做了常数平移）=====
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
    'figure.figsize': (10, 5),
    'figure.dpi': 300,
})

plt.figure()

plt.plot(sub["rel_step"], sub["chosen_s"],
         linewidth=2, label="Winning")
plt.plot(sub["rel_step"], sub["rejected_s_shifted"],
         linewidth=2, linestyle="--", label="Losing")

# 起点（严格重合）
plt.scatter(0, sub["chosen_s"].iloc[0],
            marker='*', s=220, edgecolor='black', zorder=5)

plt.xlabel("Training Step", fontweight="bold")
plt.ylabel(r"Likelihood $log\pi_{\theta}(y|x)$", fontweight="bold")
plt.ylim(-45, -35)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("closest_point_50_smoothed4.png")
