#!/bin/bash
ROOT_DIR="/workspace/wxd/LLaVA-NeXT/"

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT=$1
CONV_MODE=$2
FRAMES=$3
POOL_STRIDE=$4
POOL_MODE=$5
NEWLINE_POSITION=$6
OVERWRITE=$7
SAVE_NAME=$8
RESOLUTION=$9

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi

echo $RESOLUTION
    
python3 playground/demo/eval_dream-1k.py \
    --model-path $CKPT \
    --output_dir ./work_dirs/dream/$SAVE_DIR \
    --output_name test \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --answers-file results/answer-dream-1k-${SAVE_NAME}.json \
    --image_resolution $RESOLUTION

# llava-video
# DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_DREAM-1k.sh /volsparse3/wxd/models/qwen/LLaVA-Video-7B-Qwen2 qwen_1_5 16 1 bilinear grid True llava-video-f16 384

# # llava-video-ours
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_DREAM-1k.sh /volsparse3/wxd/ckpt/llava-video-jf-4A100/llava-ov-qwen_ours_hound-8k_f16_blinear2/checkpoint-500 qwen_1_5 16 1 bilinear grid True llava-video-f16-ours 384

# llava-video-mc-13k-ours
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_DREAM-1k.sh /volsparse3/wxd/ckpt/llava-video-jf-4A100/llava-ov-qwen_ours_hound-8k_f16_blinear2-mc-13k/checkpoint-1000 qwen_1_5 16 1 bilinear grid True llava-video-f16-mc13k-ours 384


# llava-next
# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=1 bash scripts/video/eval/eval_mlvu.sh /volsparse3/wxd/models/vicuna/LLaVA-NeXT-Video-7B vicuna_v1 16 2 average no_token True llava-next-f16 336

# export DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=3 bash scripts/video/eval/eval_mlvu.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/vicuna/llava_simpo_17k_debate-hound-ours-plus-caption vicuna_v1 16 2 average no_token True llava-next-f16-25k-data-f16 336