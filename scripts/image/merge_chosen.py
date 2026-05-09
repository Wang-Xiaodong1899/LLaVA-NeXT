import json


# jsonl_file_path = r"C:\Users\wangxiaodong\Desktop\ov-ov-inject\1201\llava_rlhf_for_dpo_ov_multi-turn_all.jsonl"
jsonl_file_path = r"C:\Users\wangxiaodong\Desktop\inject_SI-7b-1209\llava_rlhf_for_dpo_ov-si_debate_1208_0_10000.jsonl"

chosen_data = []
with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)  # Parse the JSON line into a dictionary
        chosen_data.append(entry)  # Add matching entry to the list

# jsonl_file_path = r"C:\Users\wangxiaodong\Desktop\ov-ov-inject\aug-s224-sample-1203\p0.6\llava_rlhf_for_dpo_ov_aug_s224_p0.6_0_10000.jsonl"
jsonl_file_path = r"C:\Users\wangxiaodong\Desktop\inject_SI-7b-1209\llava_rlhf_for_dpo_si_aug_s224_p0.6_0_10000.jsonl"

rejected_data = []
with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)  # Parse the JSON line into a dictionary
        rejected_data.append(entry)  # Add matching entry to the list

data_length = len(rejected_data)

time = 0
new_data = []
for (item1, item2) in zip(chosen_data, rejected_data[:data_length]):
    if item1["id"] == item2["id"]:
        item1["rejected"] = item2["rejected"]
        time = time + 1
        new_data.append(item1)

with open(r'C:\Users\wangxiaodong\Desktop\inject_SI-7b-1209\llava_rlhf_for_dpo_si_debate-chosen_aug-s224-rejected-sample0.6.jsonl', 'w', encoding='utf-8') as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()

print(len(new_data))
print(time)