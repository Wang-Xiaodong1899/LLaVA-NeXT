import os
import json

# Define the input folder path and output file path
input_folder = r'C:\Users\wangxiaodong\Desktop\checkpoint-500_vicuna_v1_frames_4_stride_2'
output_file = 'next-dpo-7b-iter2-f4_8000_16000.jsonl'

# Create the output file
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Iterate through each file in the folder
    filenames = list(os.listdir(input_folder))
    filenames.sort()
    print(filenames)
    for filename in filenames:
        if filename.endswith('.jsonl') and filename.startswith('next'):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

print("Merge complete!")
