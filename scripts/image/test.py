import json


jsonl_file_path = r"C:\Users\wangxiaodong\Downloads\rlhf-v_for_dpo_ov_debate_1205_0_6000.jsonl"

jsonl_file_path1 = r"C:\Users\wangxiaodong\Downloads\rlhf-v_for_dpo_ov_debate_1205_0_6000_fix.jsonl"

chosen_data = []
with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)  # Parse the JSON line into a dictionary
        entry["image"] = entry["image_path"]
        chosen_data.append(entry)  # Add matching entry to the list

with open(jsonl_file_path1, 'w', encoding='utf-8') as f:
    for item in chosen_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()
