import json
import os

matched_data = []

jsonl_file_path = "/volsparse1/wxd/data/llava_hound/shareVideoGPTV/filtered_video_id_1101.jsonl"

# Read the JSONL file
with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)  # Parse the JSON line into a dictionary
        matched_data.append(entry)  # Add matching entry to the list

video_dict = {}

for idx, item in enumerate(matched_data):
    if item['video'] in video_dict:
        video_dict[item['video']].append(idx)
    else:
        video_dict[item['video']] = [idx]

import random
random.seed(24)
# random choose
selected_meta = []
for k, v in video_dict.items():
    chosen_v = random.choice(v)
    selected_meta.append(matched_data[chosen_v])

with open("/volsparse1/wxd/data/llava_hound/shareVideoGPTV/filtered_video_id_random_1_1101.jsonl", 'w', encoding='utf-8') as output_file:
    for item in selected_meta:
        json.dump(item, output_file)  # Write the JSON entry to the file
        output_file.write('\n')  # Add a newline after each entry

print(f"Kept {len(selected_meta)} entries.")  # Print the count of kept entries
