import json

with open(r'C:\Users\wangxiaodong\Desktop\CVPR\MLMM\inject-1101\long_video_ge_64_qa.jsonl', 'r') as f:
    data = json.load(f)
print(len(data))
print(data[0])