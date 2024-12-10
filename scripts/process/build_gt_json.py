import os
import json

input_file_1 = "/Users/xiaodong/Downloads/sft_dpo_17k.jsonl"

chosen = []
back = []

with open(input_file_1, 'r', encoding='utf-8') as infile:
    for line in infile:
        entry = json.loads(line)
        entry["chosen"] = entry["answer"]
        chosen.append(entry)


output_file = "/Users/xiaodong/Downloads/sft_dpo_17k_gt_as_chosen.jsonl"

with open(output_file, 'w', encoding='utf-8') as outfile:
    for line in chosen:
        json.dump(line, outfile)  # Write the JSON entry to the file
        outfile.write('\n')  # Add a newline after each entry

print(chosen[0])
print("completed!")
