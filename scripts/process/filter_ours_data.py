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

# rejected_path = f"C:\\Users\\wangxiaodong\\Desktop\\reference_rejected_logps_7B.npy"
# all_chosen_path = f"C:\\Users\\wangxiaodong\\Desktop\\reference_chosen_logps_7B.npy"
# jsonl_path = f"C:\\Users\\wangxiaodong\\Desktop\\sft_dpo_17k.jsonl"

# data_path = f"C:\\Users\\wangxiaodong\\Desktop\\reference_model-ouput_logps_7B_test.npy"

chosen_path = f"/Users/xiaodong/Downloads/next-7b-f16-s2-debate-aug-f2-s3-0_17000_logp_chosen.npy"
rejected_path = f"/Users/xiaodong/Downloads/next-7b-f16-s2-debate-aug-f2-s3-0_17000_logp_rejected.npy"

rejected_data = np.load(rejected_path)
# all_chosen_data = np.load(all_chosen_path)
chosen_data = np.load(chosen_path)
# model_output = np.load(model_output_path)
# data = np.load(data_path)

gap = chosen_data - rejected_data

neg_index = np.where(gap > 0)[0]


# plt.plot(range(len(chosen_data)), chosen_data, label="chosen", c='r')
# plt.plot(range(len(rejected_data)), rejected_data, label="rejected", c='b')
# plt.axhline(y=np.mean(chosen_data), color='r', linestyle='--')
# plt.axhline(y=np.mean(rejected_data), color='b', linestyle='--')
# plt.legend()

print('chosen mean', np.mean(chosen_data))
print('rejected mean', np.mean(rejected_data))

print(f'All samples: {len(rejected_data)}')

neg_gap = gap[neg_index]
plt.title('(refer_logp_chosen-refer_logp_rejected)')
plt.plot(range(len(neg_gap)), neg_gap)

print(f'selected samples: {len(neg_gap)}')

# plt.plot(range(len(gap)), gap)
# print(f"max: {max(gap)}, min: {min(gap)}")
# print(f"max: {max(neg_gap)}, min: {min(neg_gap)}")
# plt.ylim((-2, 2))


plt.show()