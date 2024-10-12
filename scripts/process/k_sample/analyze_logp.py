import json
import matplotlib.pyplot as plt
import numpy as np
import random


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')



k1_path = r"C:\Users\wangxiaodong\Desktop\video_ov-7b-sample-K5\llava-onevision-qwen2-7b-ov_qwen_1_5_frames_16_stride_1\ov-7b_f16_K5_0_2000_k0_k1_logp_chosen.npy"
k2_path = r"C:\Users\wangxiaodong\Desktop\video_ov-7b-sample-K5\llava-onevision-qwen2-7b-ov_qwen_1_5_frames_16_stride_1\ov-7b_f16_K5_0_2000_k0_k1_logp_rejected.npy"

# rejected_data = np.load(rejected_path)
# all_chosen_data = np.load(all_chosen_path)
k1_data = np.load(k1_path)
k2_data = np.load(k2_path)


# print(len(annotation))
# print(len(annotation_neg))
plt.plot(range(len(k1_data)), k1_data, label="k1", c='r')
# plt.plot(range(len(k1_data)), k1_data, label="k2", c='b')

plt.legend()

print(np.mean(k1_data))
print(np.mean(k2_data))


plt.show()