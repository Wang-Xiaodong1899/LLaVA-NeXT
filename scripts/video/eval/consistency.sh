#!/bin/bash

CKPT=$1
SAVE_NAME=$2
FRAMES=$3
#eval_frame: 16 (align with finetuning)

bash scripts/video/eval/video_consistency.sh $CKPT vicuna_v1 $FRAMES 2 average no_token True $SAVE_NAME

python llava/eval/evaluate/evaluate_benchmark_5_consistency.py \
    --pred_path results/answer-video-consistency-${SAVE_NAME}.jsonl \
    --output_dir results/consistency_${SAVE_NAME} \
    --output_json results/review-video-consistency-${SAVE_NAME}.jsonl \
    --api_key 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 \
    --num_tasks 1