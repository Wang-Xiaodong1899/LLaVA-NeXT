import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# 读取CSV文件
data = pd.read_csv(r'C:\Users\wangxiaodong\projects\LLaVA-NeXT\simo_ours_8k_logp_visulization.csv')

# 打印表头
print("表头:", data.columns.tolist())

# 提取特定列
selected_columns = data[-200:][['train/logps/answer', 'train/logps/chosen', 'train/logps/rejected', '_step']]

# 获取数据
logp_answer = selected_columns["train/logps/answer"]
log_chosen = selected_columns["train/logps/chosen"]
log_rej = selected_columns["train/logps/rejected"]  # 如果这是重复的，可能需要确认是否是不同的列

# 平滑处理
sigma = 20  # 调整此值以控制平滑程度
smoothed_logp_answer = gaussian_filter1d(logp_answer, sigma=sigma)
smoothed_log_chosen = gaussian_filter1d(log_chosen, sigma=sigma)
smoothed_log_rejcted = gaussian_filter1d(log_rej, sigma=sigma)

plt.rcParams.update({'axes.labelsize': 15})

# 绘制曲线
plt.figure(figsize=(8, 6))
plt.plot(range(200), logp_answer, color='blue', alpha=0.1)
plt.plot(range(200), log_chosen, color='skyblue', alpha=0.3)
plt.plot(range(200), log_rej, color='green',  alpha=0.1)

plt.plot(range(200), smoothed_logp_answer, label='Answer (GT)', color='blue')
plt.plot(range(200), smoothed_log_chosen, label='Chosen (model output, w/ prior)', color='skyblue')
plt.plot(range(200), smoothed_log_rejcted, label='Rejected (model output, w/o prior)', color='green')  # 确认是否需要不同的数据

# 添加标题和标签
# plt.title('LogP Visualization')
plt.xlim(0, 200)
plt.xlabel('Sample index')
plt.ylabel('Log prob. (reward)')
plt.legend(fontsize=15)
plt.grid()

# 显示图形
plt.show()