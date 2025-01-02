#!/bin/bash


base_url="https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/resolve/main/2_3_m_youtube_v0_1/2_3_m_youtube_v0_1_videos_"

start=1
end=50


for i in $(seq $start $end)
do
    file_name="2_3_m_youtube_v0_1_videos_${i}.tar.gz"
    url="${base_url}${i}.tar.gz"

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
