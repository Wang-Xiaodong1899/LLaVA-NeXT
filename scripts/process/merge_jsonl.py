import os
import json

# Define the input folder path and output file path
input_folder = "/mnt/bn/wxd-video-understanding/wangxd/LLaVA-NeXT/data/nohallu/chosen-hound-0514/LLaVA-NeXT-Video-7B_vicuna_v1_frames_16_stride_2/"
output_file = "/mnt/bn/wxd-video-understanding/wangxd/LLaVA-NeXT/data/nohallu/chosen-hound-0514/next-7b-f16-s2-videoinput-17k.jsonl"

# Create the output file
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Iterate through each file in the folder
    filenames = list(os.listdir(input_folder))
    filenames.sort()
    print(filenames)
    for filename in filenames:
        if filename.endswith('.jsonl'):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

print("Merge complete!")
