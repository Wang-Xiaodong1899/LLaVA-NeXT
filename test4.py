import os
import json

# Define the input folder path and output file path
input_folder = r'C:\Users\wangxiaodong\Desktop\ov-ov-inject'
filename = 'llava_rlhf_for_dpo_ov_debate_1201_0_10000.jsonl'
output_file = 'llava_rlhf_for_dpo_ov_debate_1201_0_10000_fix.jsonl'

# Create the output file
with open(os.path.join(input_folder, output_file), 'w', encoding='utf-8') as outfile:
    with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.replace("rejeted", "rejected")
            outfile.write(line)

print("replace complete!")
