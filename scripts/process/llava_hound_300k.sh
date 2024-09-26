#!/bin/bash


base_url="https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/train_300k/"


start=0
end=15

cd /volsparse1/wxd/data/llava_hound

for i in $(seq $start $end)
do
    file_name="chunk_${i}.tar.gz"
    url="${base_url}chunk_${i}.tar.gz"

    echo "Downloading ${file_name}..."

    wget --continue --retry-connrefused --waitretry=5 --tries=5 --timeout=30 "${url}"


    if [ $? -eq 0 ]; then
        echo "${file_name} downloaded successfully."
    else
        echo "Failed to download ${file_name}. Retrying..."

        wget --continue --retry-connrefused --waitretry=5 --tries=5 --timeout=30 "${url}"
        if [ $? -eq 0 ]; then
            echo "${file_name} downloaded successfully after retry."
        else
            echo "Failed to download ${file_name} after multiple attempts."
        fi
    fi
done

echo "Download completed."
