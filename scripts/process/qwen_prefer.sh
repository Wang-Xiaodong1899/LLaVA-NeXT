#!/bin/bash

CKPT=$1
VIDEO_PATH=$2
JSONLFILE=$3
START=$4
END=$5


python3 scripts/process/qwen_generate_preference.py \
    --model-path $CKPT \
    --video_root ${VIDEO_PATH} \
    --output_dir /mnt/bn/wxd-video-understanding/wangxd/LLaVA-NeXT/data/qwen-debate-hound-17k/qwen-hound-17k-0518/ \
    --output_name Qwen25-7b-f2-s2-${START}_${END} \
    --jsonl-file $JSONLFILE \
    --start $START \
    --end $END \


# Next
# CUDA_VISIBLE_DEVICES=6 bash scripts/process/qwen_prefer.sh /mnt/bn/wxd-video-understanding/wangxd/models/Qwen2.5-VL-7B-Instruct/ /mnt/bn/wxd-video-understanding/wangxd/data/shareVideoGPTV/dpo_train_data /mnt/bn/wxd-video-understanding/wangxd/data/shareVideoGPTV/sft_dpo_17k.jsonl 16000 18000
