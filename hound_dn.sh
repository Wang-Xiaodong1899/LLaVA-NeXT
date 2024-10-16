#!/bin/bash

cd /root/autodl-tmp/data/llava_hound

# Base URL
base_url="https://hf-mirror.com/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/train_300k/"

tar -zxvf chunk_0.tar.gz -C data
rm chunk_0.tar.gz
tar -zxvf chunk_1.tar.gz -C data
rm chunk_1.tar.gz
tar -zxvf chunk_2.tar.gz -C data
rm chunk_2.tar.gz
tar -zxvf chunk_3.tat.gz -C data
rm chunk_3.tar.gz
tar -zxvf chunk_4.tar.gz -C data
rm chunk_4.tar.gz


# Loop to download files from chunk_0.tar.gz to chunk_10.tar.gz
for i in {5..10}; do
    file="chunk_${i}.tar.gz"
    url="${base_url}${file}"

    # Download the file with wget, and retry once if it fails
    wget --retry-connrefused --tries=2 "${url}"

    # Check if the file downloaded successfully
    if [[ $? -ne 0 ]]; then
        echo "Download failed for ${file}, retrying once..."
        wget --retry-connrefused --tries=2 "${url}"
        
        # Check again if the download was successful after retry
        if [[ $? -ne 0 ]]; then
            echo "Failed to download ${file} after retry."
        else
            echo "${file} downloaded successfully after retry."
        fi
    else
        echo "${file} downloaded successfully."
        tar -zxvf ${file} -C data
        rm ${file}
    fi
done
