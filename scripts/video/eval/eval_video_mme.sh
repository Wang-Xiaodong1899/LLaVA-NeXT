#!/bin/bash

CKPT=$1
SAVE_NAME=$2
FRAMES=$3
DURATION=$4 # short
OVERWRITE=${5:False} # overwrite previous eval result

CONV_MODE=$6
POOL_STRIDE=$7
POOL_MODE=$8
NEWLINE_POSITION=$9
RESOLUTION=${10:336}

#eval_frame: 16 (align with finetuning)
if [ "$OVERWRITE" = True ]; then
    bash scripts/video/eval/video_mme.sh $CKPT $CONV_MODE $FRAMES $POOL_STRIDE $POOL_MODE $NEWLINE_POSITION True $SAVE_NAME $DURATION $RESOLUTION
fi

python playground/demo/eval_video_mme.py \
    --results_file results/answer-video-mme-${SAVE_NAME}.json  \
    --video_duration_type $DURATION \
    --return_categories_accuracy

# tip

# vicuna: llava-next-video
# export DECORD_EOF_RETRY_MAX=20480 && CUDA_VISIBLE_DEVICES=1 bash scripts/video/eval/eval_video_mme.sh ckpt_path name-medium 16 medium True vicuna_v1 2 average no_token

# qwen: onevision
# export DECORD_EOF_RETRY_MAX=20480 && CUDA_VISIBLE_DEVICES=1 bash scripts/video/eval/eval_video_mme.sh /workspace/wxd/LLaVA-NeXT/qwen/llava-onevision-qwen2-7b-ov 7b-ov-short 16 short True qwen_1_5 1 bilinear one_token
