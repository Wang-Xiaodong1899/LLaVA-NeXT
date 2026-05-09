import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import colorsys

def lighten_color(color, amount=0.5):
    """
    根据给定的颜色生成一个浅色版本。
    参数：
      color: 颜色名称、十六进制字符串或 RGB 元组
      amount: 浅化程度（0-1之间），值越大越接近白色
    """
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    lightened = colorsys.hls_to_rgb(c[0], 1 - amount*(1 - c[1]), c[2])
    return lightened

# 数据设置
data_sizes = ['5k', '10k', '17k', '25k', '30k']
x = np.arange(len(data_sizes))

# 主方法在不同数据量下的性能（示例数据）
performance_short = [ 45.2, 46.0, 48.6, 51.1, 50.6]
performance_medium = [ 41.6, 41.2, 42.2, 42.3, 43.3]
performance_long = [ 34.7, 35.6, 35.6, 36.0, 35.4]

# baseline 方法的性能（与数据量无关）
baseline_short = 44.0
baseline_medium = 38.0
baseline_long = 34.4

# 使用适合论文的绘图风格
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(10, 5))

# 绘制主方法的曲线，采用不同颜色和标记
ax.plot(x, performance_short, marker='o', color='tab:blue', linestyle='-', linewidth=2, markersize=8, label='Short')
ax.plot(x, performance_medium, marker='s', color='tab:red', linestyle='-', linewidth=2, markersize=8, label='Medium')
ax.plot(x, performance_long, marker='D', color='tab:purple', linestyle='-', linewidth=2, markersize=8, label='Long')

# 生成对应颜色的浅色，用于绘制 baseline 的虚线
light_blue = lighten_color('tab:blue', amount=0.5)
light_red = lighten_color('tab:red', amount=0.5)
light_purple = lighten_color('tab:purple', amount=0.5)

# 绘制 baseline 的水平虚线
ax.axhline(y=baseline_short, color=light_blue, linestyle='--', linewidth=2)
ax.axhline(y=baseline_medium, color=light_red, linestyle='--', linewidth=2)
ax.axhline(y=baseline_long, color=light_purple, linestyle='--', linewidth=2)

# 设置 x 轴刻度和标签
ax.set_xticks(x)
ax.set_xticklabels(data_sizes, fontsize=20)
ax.set_xlabel('Dataset Size', fontsize=24)
ax.set_ylabel('Acc. on Video-MME (%)', fontsize=24)

# 添加图例
ax.legend(fontsize=18, loc='best', frameon=True)

plt.tight_layout()

# 保存图形为 PDF 文件
plt.savefig('mme_size.pdf', format='pdf')
# plt.show()
