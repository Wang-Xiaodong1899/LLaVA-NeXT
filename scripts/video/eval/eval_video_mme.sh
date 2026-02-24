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

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.5-simpomargin0.5/checkpoint-2000 ls0.5-ours-17k-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.3-simpomargin0.5/checkpoint-500 ls0.3-ours-17k-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-hallutext/checkpoint-2000 hound-win-text-hallu-rej-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-hound-chosen-hound-rej-train/checkpoint-1000 hound-chosen-hound-rej-1k-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-hound-gtchosen-hound-rej-train/checkpoint-1000 hound-gt-chosen-hound-rej-1k-short 16 short True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-texthallu-rej-train-linear/checkpoint-1500 our-chosen-text-hallu-rej-1500-short 16 short True vicuna_v1 2 average no_token 336

# new
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.8-simpomargin0.5-our-chosen-our-rej-train/checkpoint-500/ ls0.8-ours-17k-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.9-simpomargin0.5-our-chosen-our-rej-train/checkpoint-500/ ls0.9-ours-17k-short 16 short True vicuna_v1 2 average no_token 336

# d
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.1/checkpoint-250/ ls0.9-ours-17k-d0.1-short 16 short True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.1/checkpoint-250/ ls0.9-ours-17k-d0.1-medium 16 medium True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=3 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.1/checkpoint-250/ ls0.9-ours-17k-d0.1-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=1 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.3/checkpoint-250/ ls0.9-ours-17k-d0.3-short 16 short True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=4 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.3/checkpoint-250/ ls0.9-ours-17k-d0.3-medium 16 medium True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=5 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.3/checkpoint-250/ ls0.9-ours-17k-d0.3-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.5/checkpoint-250/ ls0.9-ours-17k-d0.5-short 16 short True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=1 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.5/checkpoint-250/ ls0.9-ours-17k-d0.5-medium 16 medium True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.5/checkpoint-250/ ls0.9-ours-17k-d0.5-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=3 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.7/checkpoint-250/ ls0.9-ours-17k-d0.7-short 16 short True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=4 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.7/checkpoint-250/ ls0.9-ours-17k-d0.7-medium 16 medium True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=5 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.7/checkpoint-250/ ls0.9-ours-17k-d0.7-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.9/checkpoint-250/ ls0.9-ours-17k-d0.9-short 16 short True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=1 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.9/checkpoint-250/ ls0.9-ours-17k-d0.9-medium 16 medium True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d0.9/checkpoint-250/ ls0.9-ours-17k-d0.9-long 16 long True vicuna_v1 2 average no_token 336

# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=3 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d1.0/checkpoint-250/ ls0.9-ours-17k-d1.0-short 16 short True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=4 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d1.0/checkpoint-250/ ls0.9-ours-17k-d1.0-medium 16 medium True vicuna_v1 2 average no_token 336
# DECORD_EOF_RETRY_MAX=40960 CUDA_VISIBLE_DEVICES=5 bash scripts/video/eval/eval_video_mme.sh /mnt/bn/wxd-video-understanding/wangxd/ckpt/llava-next-8H20/llava_simpo_17k_debate-hound-17k-dynalabelsmooth-pilog-0-ls0.1-simpomargin0.5-our-chosen-our-rej-train-d1.0/checkpoint-250/ ls0.9-ours-17k-d1.0-long 16 long True vicuna_v1 2 average no_token 336



# eval process
# cd /mnt/bn/wxd-video-understanding/wangxd/LLaVA-NeXT
# . videoenv/bin/activate


# qwen: onevision
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh /volsparse2/wxd/ckpt/llava-ov-jf-4A100/llava_qwen_simpo_inject_prior_aug-f2-8k-stride_3/checkpoint-400/ 7b-ov-stride3-short 32 short True qwen_1_5 1 bilinear one_token 384
