import os
import json
from tqdm import tqdm

# 设置文件夹路径和输出文件路径
folder_path = '/volsparse1/wxd/data/llava_hound/shareVideoGPTV/QA'  # 请替换为你的文件夹路径
output_file = 'long_video_ge_64.jsonl'

# 获取一级子文件夹列表
entries = os.listdir(folder_path)

# 打开输出文件以写入模式
with open(output_file, 'w') as f:
    # 遍历一级子文件夹，添加进度条
    for entry in tqdm(entries, desc="Processing folders"):
        full_path = os.path.join(folder_path, entry)
        if os.path.isdir(full_path):  # 确保是子文件夹
            file_count = len(os.listdir(full_path))
            if file_count >= 64:
                # 写入符合条件的子文件夹名称到 JSONL 文件
                f.write(json.dumps({"folder_name": entry}) + '\n')

print(f"满足条件的子文件夹已保存到 {output_file}")
