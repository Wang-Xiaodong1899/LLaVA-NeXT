import os
import json

# Define the input folder path and output file path
input_folder = "/volsparse3/wxd/data/self-gen/llava-next-aug-video-178k/2_3_m_youtube_v0_1-2_3_m_youtube_oe-0110/LLaVA-NeXT-Video-7B_vicuna_v1_frames_1_stride_2/"
output_file = "/volsparse3/wxd/data/self-gen/llava-next-aug-video-178k/2_3_m_youtube_v0_1-2_3_m_youtube_oe-0110/llava-next-7b-f2-s2-0_16000.jsonl"

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
