#!/bin/bash

# Set base URL and target directory
base_url="https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/train_600k"
target_dir="/volsparse1/wxd/data/llava_hound/caption"

# Create the target directory if it doesn't exist
mkdir -p "${target_dir}"

# Loop to download files from chunk_0 to chunk_5
for i in {0..5}; do
    # Construct filename and complete URL
    file_name="chunk_${i}.tar.gz"
    url="${base_url}/${file_name}"

    # Download the file to the target directory with automatic retries if interrupted
    wget -c -O "${target_dir}/${file_name}" "${url}"
done

echo "Download completed and saved to ${target_dir}!"
