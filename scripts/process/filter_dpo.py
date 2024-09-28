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
jsonl_path = f"C:\\Users\\wangxiaodong\\Desktop\\sft_dpo_17k.jsonl"

data_path = f"C:\\Users\\wangxiaodong\\Desktop\\reference_model-ouput_logps_7B_test.npy"

chosen_path = f"C:\\Users\\wangxiaodong\\Desktop\\reference_chosen_logps_7B-DPO-2k_test.npy"
model_output_path = f"C:\\Users\\wangxiaodong\\Desktop\\reference_model-ouput_logps_7B-DPO-2k_test.npy"

# rejected_data = np.load(rejected_path)
# all_chosen_data = np.load(all_chosen_path)
chosen_data = np.load(chosen_path)
model_output = np.load(model_output_path)
data = np.load(data_path)

# gap = chosen_data - rejected_data

# neg_index = np.where(gap < 0)[0]


annotation = load_jsonl(jsonl_path)

print(len(annotation))

jsonl_pos_path = f"C:\\Users\\wangxiaodong\\Desktop\\sft_dpo_17k_filter_neg.jsonl"

pos_annotation = load_jsonl(jsonl_pos_path)

random_idx = random.sample(range(len(annotation)), len(pos_annotation))

random_annotation = [annotation[idx] for idx in random_idx]

jsonl_rand_path = f"C:\\Users\\wangxiaodong\\Desktop\\sft_dpo_17k_filter_rand_len-pos.jsonl"

# write_jsonl(jsonl_rand_path, random_annotation)

# annotation_neg = [annotation[idx] for idx in neg_index]



# write_jsonl(jsonl_pos_path, annotation_neg)

# print(len(annotation))
# print(len(annotation_neg))
plt.plot(range(len(data)), chosen_data[:len(data)], label="chosen", c='r')
plt.plot(range(len(data)), model_output[:len(data)], label="model-output", c='b')
# plt.plot(range(len(chosen_data)), all_chosen_data[:len(chosen_data)], label="all_chosen", c='gold')

plt.axhline(y=np.mean(chosen_data[:len(data)]), color='r', linestyle='--')
plt.axhline(y=np.mean(model_output[:len(data)]), color='b', linestyle='--')
# plt.axhline(y=np.mean(all_chosen_data[:len(chosen_data)]), color='gold', linestyle='--')

plt.legend()

print(np.mean(chosen_data[:len(data)]))
print(np.mean(model_output[:len(data)]))
# print(np.mean(all_chosen_data[:len(chosen_data)]))

# neg_gap = gap[neg_index]
# plt.title('(refer_logp_chosen-refer_logp_rejected)')
# plt.plot(range(len(neg_gap)), neg_gap)

# plt.plot(range(len(gap)), gap)
# print(f"max: {max(gap)}, min: {min(gap)}")
# print(f"max: {max(neg_gap)}, min: {min(neg_gap)}")
plt.ylim((-100, 0))


plt.show()