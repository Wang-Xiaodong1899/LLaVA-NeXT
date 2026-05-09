#!/bin/bash


# CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/temporal.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/llava_simpo_17k_debate-hound-ours-sft simpo_17k_debate-hound-ours-sft-f16 16 2
# CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/temporal.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/vicuna/llava_simpo_17k_debate-hound-ours-sft-caption-8k-epoch2 ours-iter2-caption-8k-ep2-f16 16 2
# CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/temporal.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/vicuna/llava_simpo_17k_debate-hound-ours-plus-caption ours-plus-caption-f16 16 2

CKPT=$1
SAVE_NAME=$2
FRAMES=$3
STRIDE=${4:-2}
INFER=${5:-False}

#eval_frame: 16 (align with finetuning)


# vicuna
if [ "$INFER" = True ]; then
    bash scripts/video/eval/video_temporal.sh $CKPT vicuna_v1 $FRAMES $STRIDE average no_token True $SAVE_NAME
fi

# ov
# bash scripts/video/eval/video_temporal.sh $CKPT qwen_1_5 $FRAMES $STRIDE bilinear one_token True $SAVE_NAME

python3 llava/eval/evaluate/evaluate_benchmark_4_temporal.py \
    --pred_path results/answer-video-temporal-${SAVE_NAME}.jsonl \
    --output_dir results/temporal_${SAVE_NAME}_0125 \
    --output_json results/review-video-temporal-${SAVE_NAME}_0125.jsonl \
    --api_key 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 \
    --num_tasks 1