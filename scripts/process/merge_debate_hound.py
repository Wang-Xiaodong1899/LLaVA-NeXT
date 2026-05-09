import os
import json

input_file_1 = r"C:\Users\wangxiaodong\Desktop\debate-hound-1211\next-7b-f16-s2-0_8000.jsonl"
input_file_2 = r"C:\Users\wangxiaodong\Desktop\sft_dpo_17k.jsonl"

chosen = []
back = []

with open(input_file_1, 'r', encoding='utf-8') as infile:
    for line in infile:
        entry = json.loads(line)
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

output_file = r"C:\Users\wangxiaodong\Desktop\debate-hound-1211\next-7b-f16-s2-hound-rej-0_8000.jsonl"

with open(output_file, 'w', encoding='utf-8') as outfile:
    for line in chosen:
        json.dump(line, outfile)  # Write the JSON entry to the file
        outfile.write('\n')  # Add a newline after each entry

print(chosen[0])
print("completed!")
