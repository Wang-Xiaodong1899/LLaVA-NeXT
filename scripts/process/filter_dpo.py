import json
import matplotlib.pyplot as plt
import numpy as np


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

rejected_path = f"C:\\Users\\wangxiaodong\\Desktop\\reference_rejected_logps_7B.npy"
chosen_path = f"C:\\Users\\wangxiaodong\\Desktop\\reference_chosen_logps_7B.npy"
jsonl_path = f"C:\\Users\\wangxiaodong\\Desktop\\sft_dpo_17k.jsonl"

rejected_data = np.load(rejected_path)
chosen_data = np.load(chosen_path)

gap = chosen_data - rejected_data

pos_index = np.where(gap > 0)[0]


annotation = load_jsonl(jsonl_path)

annotation_pos = [annotation[idx] for idx in pos_index]

jsonl_pos_path = f"C:\\Users\\wangxiaodong\\Desktop\\sft_dpo_17k_filter_pos.jsonl"

write_jsonl(jsonl_pos_path, annotation_pos)

print(len(annotation))
print(len(annotation_pos))

pos_gap = gap[pos_index]
# plt.plot(range(len(pos_gap)), pos_gap)

# plt.plot(range(len(gap)), gap)
print(f"max: {max(gap)}, min: {min(gap)}")
print(f"max: {max(pos_gap)}, min: {min(pos_gap)}")



plt.show()