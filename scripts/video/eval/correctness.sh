#!/bin/bash

CKPT=$1
SAVE_NAME=$2

# bash scripts/video/demo/video_generic.sh /mnt/storage/user/wangxiaodong/LLaVA-NeXT/vicuna/LLaVA-NeXT-Video-7B-DPO vicuna_v1 32 2 average no_token True LLaVA-NeXT-Video-7B-DPO

python llava/eval/evaluate/evaluate_benchmark_1_correctness.py \
    --pred_path results/answer-video-generic-${SAVE_NAME}.jsonl \
    --output_dir results/correctness_${SAVE_NAME} \
    --output_json results/review-video-correctness-${SAVE_NAME}.jsonl \
    --api_key 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 \
    --num_tasks 1