import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# 读取CSV文件
data = pd.read_csv('/Users/xiaodong/MLM+RL/LLaVA-NeXT/simpo_dpo_16k_logp_visulization.csv')

# 打印表头
print("表头:", data.columns.tolist())

# 提取特定列
selected_columns = data[:200][['train/logps/answer', 'train/logps/chosen', 'train/logps/rejected', '_step']]

# 获取数据
logp_answer = selected_columns["train/logps/answer"]
log_chosen = selected_columns["train/logps/chosen"]
log_rej = selected_columns["train/logps/rejected"]  # 如果这是重复的，可能需要确认是否是不同的列

# 平滑处理
sigma = 20  # 调整此值以控制平滑程度
# smoothed_logp_answer = gaussian_filter1d(logp_answer, sigma=sigma)
smoothed_log_chosen = gaussian_filter1d(log_chosen, sigma=sigma)
smoothed_log_rejcted = gaussian_filter1d(log_rej, sigma=sigma)

# 绘制曲线
plt.figure(figsize=(12, 6))
# plt.plot(selected_columns['_step'], logp_answer, color='blue', alpha=0.1)
plt.plot(selected_columns['_step'], log_chosen, color='orange', alpha=0.1)
plt.plot(selected_columns['_step'], log_rej, color='green',  alpha=0.1)

# plt.plot(selected_columns['_step'], smoothed_logp_answer, label='GT Answer', color='blue')
plt.plot(selected_columns['_step'], smoothed_log_chosen, label='Chosen', color='orange')
plt.plot(selected_columns['_step'], smoothed_log_rejcted, label='Rejected', color='green')  # 确认是否需要不同的数据

# 添加标题和标签
plt.title('LogP Visualization')
plt.xlabel('Step')
plt.ylabel('Values')
plt.legend()
plt.grid()

# 显示图形
plt.show()