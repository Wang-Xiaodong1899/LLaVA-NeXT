#!/bin/bash

# Set base URL and target directories
base_url="https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/train_300k"
target_dir="/data/llava_hound/train_300k"
data_dir="${target_dir}/data"

# Create target and data directories if they don't exist
mkdir -p "${target_dir}"
mkdir -p "${data_dir}"

# Loop to download and extract files from chunk_0 to chunk_5
for i in {0..5}; do
    # Construct filename and complete URL
    file_name="chunk_${i}.tar.gz"
    url="${base_url}/${file_name}"
    file_path="${target_dir}/${file_name}"

    # Download the file to the target directory with automatic retries if interrupted
    wget -c -O "${file_path}" "${url}"

    # Extract the file to the data directory
    tar -xzvf "${file_path}" -C "${data_dir}"
done

echo "Download and extraction completed! Files are in ${data_dir}."
