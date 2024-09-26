import os
import json

# Define the input folder path and output file path
input_folder = 'volsparse1/wxd/data/self-gen/video-aug/checkpoint-3000_vicuna_v1_frames_4_stride_2'
output_file = 'aug_f4_0_8000.jsonl'

# Create the output file
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Iterate through each file in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jsonl'):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

print("Merge complete!")
