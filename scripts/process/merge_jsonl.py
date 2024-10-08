import os
import json

# Define the input folder path and output file path
input_folder = '/volsparse1/wxd/data/self-gen/video_ov-72b/llava-onevision-qwen2-72b-ov-sft_qwen_1_5_frames_32_stride_1'
output_file = 'ov-72b-f32_0_16000.jsonl'

# Create the output file
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Iterate through each file in the folder
    filenames = list(os.listdir(input_folder))
    filenames.sort()
    print(filenames)
    for filename in filenames:
        if filename.endswith('.jsonl') and filename.startswith('ov'):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

print("Merge complete!")
