#!/bin/bash

CKPT=$1
SAVE_NAME=$2
FRAMES=$3

# bash scripts/video/eval/video_generic.sh $CKPT vicuna_v1 $FRAMES 2 average no_token True $SAVE_NAME

# python llava/eval/evaluate/evaluate_benchmark_1_correctness.py \
#     --pred_path results/answer-video-generic-${SAVE_NAME}.jsonl \
#     --output_dir results/correctness_${SAVE_NAME} \
#     --output_json results/review-video-correctness-${SAVE_NAME}.jsonl \
#     --api_key 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 \
#     --num_tasks 1

# # orientation
# python llava/eval/evaluate/evaluate_benchmark_2_detailed_orientation.py \
#     --pred_path results/answer-video-generic-${SAVE_NAME}.jsonl \
#     --output_dir results/detailed_orientation_${SAVE_NAME} \
#     --output_json results/review-video-detailed_orientation-${SAVE_NAME}.jsonl \
#     --api_key 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 \
#     --num_tasks 1

# context
python llava/eval/evaluate/evaluate_benchmark_3_context.py \
    --pred_path results/answer-video-generic-${SAVE_NAME}.jsonl \
    --output_dir results/context_${SAVE_NAME} \
    --output_json results/review-video-context-${SAVE_NAME}.jsonl \
    --api_key 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 \
    --num_tasks 1

