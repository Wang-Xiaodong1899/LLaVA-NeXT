#!/bin/bash


base_url="https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main/videos_chunked_"

# NOTE if can't connect to https://huggingface.co
# change to base_url="https://hf-mirror.com/datasets/lmms-lab/Video-MME/resolve/main/videos_chunked_"


start=1
end=20


for i in $(seq -f "%02g" $start $end)
do
    file_name="videos_chunked_${i}.zip"
    url="${base_url}${i}.zip"

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
