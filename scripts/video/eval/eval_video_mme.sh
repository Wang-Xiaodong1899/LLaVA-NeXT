#!/bin/bash

CKPT=$1
SAVE_NAME=$2
FRAMES=$3
DURATION=$4 # short
OVERWRITE=$5 # overwrite previous eval result

CONV_MODE=$6
POOL_STRIDE=$7
POOL_MODE=$8
NEWLINE_POSITION=$9
RESOLUTION=${10}

#eval_frame: 16 (align with finetuning)
if [ "$OVERWRITE" = True ]; then
    bash scripts/video/eval/video_mme.sh $CKPT $CONV_MODE $FRAMES $POOL_STRIDE $POOL_MODE $NEWLINE_POSITION True $SAVE_NAME $DURATION $RESOLUTION
fi

python3 playground/demo/eval_video_mme.py \
    --results_file results/answer-video-mme-${SAVE_NAME}.json  \
    --video_duration_type $DURATION \
    --return_categories_accuracy

# tip

# vicuna: llava-next-video
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /volsparse2/wxd/ckpt/llava-next-jf-4A100/llava_vicuna_simpo_inject_8k_aug_8k_f32_stride_3/checkpoint-100/ simpo-next-f32-stride-3-medium 32 medium True vicuna_v1 2 average no_token 336

# qwen: onevision
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /volsparse2/wxd/ckpt/llava-ov-jf-4A100/llava_qwen_simpo_inject_prior_aug-f2-8k-stride_3/checkpoint-400/ 7b-ov-stride3-short 32 short True qwen_1_5 1 bilinear one_token 384
