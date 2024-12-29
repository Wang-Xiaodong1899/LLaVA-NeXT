import os
import json

input_file = "/root/private_data/data/data/shareVideoGPTV/sft_dpo_17k.jsonl"
output_file = "/root/private_data/data/data/shareVideoGPTV/sft_dpo_10k.jsonl"

iter = 0
# Create the output file
with open(output_file, 'w', encoding='utf-8') as outfile:
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                if iter >= 5000:
                    break
                outfile.write(line)
                iter = iter+1

print("save completed!")
