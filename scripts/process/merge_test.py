import os
import json

input_file_1 = "ov-72b-f32_0_8000.jsonl"
input_file_2 = "ov-7b-aug_f4_0_8000.jsonl"

chosen = []
back = []

with open(input_file_1, 'r', encoding='utf-8') as infile:
    for line in infile:
        entry = json.loads(line)
        entry["chosen"] = entry["rejected"]
        entry.pop("rejected")
        chosen.append(entry)

with open(input_file_2, 'r', encoding='utf-8') as infile:
    for line in infile:
        entry = json.loads(line)
        back.append(entry)

for idx, line in enumerate(chosen):
    # match data
    for line2 in back:
        if line2["id"] == line["id"]:
            chosen[idx]["rejected"] = line2["rejected"]

output_file = "ov-72b-f32_add_7b_aug_reject_0_8000.jsonl"

with open(output_file, 'w', encoding='utf-8') as outfile:
    for line in chosen:
        json.dump(line, output_file)  # Write the JSON entry to the file
        outfile.write('\n')  # Add a newline after each entry
print("completed!")