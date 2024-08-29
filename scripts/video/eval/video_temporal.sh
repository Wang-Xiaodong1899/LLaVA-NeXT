#!/bin/bash
ROOT_DIR="/volsparse2/wxd/projects/LLaVA-NeXT/"

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

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi
    
python3 playground/demo/video_general.py \
    --model-path $CKPT \
    --output_dir ./work_dirs/video_demo/$SAVE_DIR \
    --output_name test \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --answers-file results/answer-video-temporal-${SAVE_NAME}.jsonl

# example
# bash scripts/video/demo/video_temporal.sh /volsparse2/wxd/models/vicuna/LLaVA-NeXT-Video-7B-DPO vicuna_v1 32 2 average no_token True LLaVA-NeXT-Video-7B-DPO

# python llava/eval/evaluate/evaluate_benchmark_4_temporal.py \
# --pred_path results/answer-video-temporal.jsonl \
# --output_dir results/temporal \
# --output_json results/review-video-temporal-xxx.jsonl \
# --api_key [openai api key] \
# --num_tasks 1