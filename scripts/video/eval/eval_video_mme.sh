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
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /data/wangxd/ckpt/llava-next-PKU-4A100/llava_ours_debate-hound-17k/checkpoint-1000/ next-ours-17k-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/multimodal-datasets-hl/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.5-simpomargin0.5/checkpoint-2000 ls0.5-ours-17k-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/multimodal-datasets-hl/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.3-simpomargin0.5/checkpoint-500 ls0.3-ours-17k-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/multimodal-datasets-hl/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-hallutext/checkpoint-2000 hound-win-text-hallu-rej-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/multimodal-datasets-hl/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-hound-chosen-hound-rej-train/checkpoint-1000 hound-chosen-hound-rej-1k-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/multimodal-datasets-hl/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-hound-gtchosen-hound-rej-train/checkpoint-1000 hound-gt-chosen-hound-rej-1k-short 16 short True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/multimodal-datasets-hl/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-texthallu-rej-train-linear/checkpoint-1500 our-chosen-text-hallu-rej-1500-short 16 short True vicuna_v1 2 average no_token 336


# qwen: onevision
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /volsparse2/wxd/ckpt/llava-ov-jf-4A100/llava_qwen_simpo_inject_prior_aug-f2-8k-stride_3/checkpoint-400/ 7b-ov-stride3-short 32 short True qwen_1_5 1 bilinear one_token 384
