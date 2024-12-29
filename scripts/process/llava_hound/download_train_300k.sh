#!/bin/bash


base_url="https://hf-mirror.com/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/train_300k/chunk"

start=0
end=15


for i in $(seq -f "%g" $start $end)
do
    file_name="chunk_${i}.tar.gz"
    url="${base_url}_${i}.tar.gz"

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
