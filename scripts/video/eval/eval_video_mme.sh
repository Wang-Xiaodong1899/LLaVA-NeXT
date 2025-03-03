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

# baseline
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/models/vicuna/LLaVA-NeXT-Video-7B next-7b-medium 16 medium True vicuna_v1 2 average no_token 336

# /volsparse3/wxd/ckpt/llava-next-jf-4A100/llava_dpo-hound-17k-our-data/checkpoint-1000/

# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/llava_dpo-hound-17k-our-data/checkpoint-1000/ dpo-ours-long 16 long True vicuna_v1 2 average no_token 336


# oe-16k
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/vicuna/llava_simpo_debate-video-oe-16k next-oe-16k-long 16 long True vicuna_v1 2 average no_token 336

# vicuna: llava-next-video
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/llava_simpo_17k_debate-hound-ours-sft hound-ours-sft-short 16 short True vicuna_v1 2 average no_token 336

# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=3 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/llava_simpo_8k_inject-QA/checkpoint-250 simpo-our-QA-250-7b-medium 16 medium True vicuna_v1 2 average no_token 336

# caption
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/vicuna/llava_simpo_17k_debate-hound-ours-sft-caption-8k-epoch2 ours-iter2-caption-8k-ep2-short 16 short True vicuna_v1 2 average no_token 336

# combine
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/vicuna/llava_simpo_17k_debate-hound-ours-plus-caption ours-plus-caption-1500-long 16 long True vicuna_v1 2 average no_token 336

# dpo-5k
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/vicuna/llava_simpo_debate-hound-ours-plus-caption-qa-30k/llava_simpo_17k_debate-hound-30k-ours-sft/ ours-30k-short 16 short True vicuna_v1 2 average no_token 336


# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/llava_simpo_17k_debate-hound-ours-sft-f16-nodynamic-alpha-0210/checkpoint-375 next-ours-375-short 16 short True vicuna_v1 2 average no_token 336
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=3 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/llava_simpo_17k_debate-hound-ours-sft-f16-0210/checkpoint-500 next-ours-dynamic-500-short 16 short True vicuna_v1 2 average no_token 336


# llava-video
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=1 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/models/qwen/LLaVA-Video-7B-Qwen2  llava-video-medium 16 medium True qwen_1_5 1 bilinear grid 384

# llava-video-ours
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-video-jf-4A100/llava-ov-qwen_ours_hound-8k_f16_blinear2/checkpoint-500 llava-video-ours-long 16 long True qwen_1_5 1 bilinear grid 384

# oe-13k
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=1 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-video-jf-4A100/llava-ov-qwen_ours_hound-8k_f16_blinear2-oe-13k/checkpoint-1000 llava-video-ours-oe13k-short 16 short True qwen_1_5 1 bilinear grid 384

# dynamic
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /volsparse3/wxd/ckpt/llava-video-jf-4A100/llava-ov-qwen_ours_hound-8k_f16_blinear2-dynamic/checkpoint-500 llava-video-ours-dyna-short 16 short True qwen_1_5 1 bilinear grid 384



# qwen: onevision
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /volsparse2/wxd/ckpt/llava-ov-jf-4A100/llava_qwen_simpo_inject_prior_aug-f2-8k-stride_3/checkpoint-400/ 7b-ov-stride3-short 32 short True qwen_1_5 1 bilinear one_token 384
