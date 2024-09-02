#!/bin/bash
ROOT_DIR="/home/wxd/projects/LLaVA-NeXT/"

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
    --question-file /home/wxd/projects/LLaVA-NeXT/llava/eval/questions/video_qa/generic_qa.json \
    --answers-file results/answer-video-generic-${SAVE_NAME}.jsonl

# example
# bash scripts/video/eval/video_generic.sh /home/wxd/projects/LLaVA-NeXT/vicuna/LLaVA-NeXT-Video-7B-DPO vicuna_v1 32 2 average no_token True LLaVA-NeXT-Video-7B-DPO

# python llava/eval/evaluate/evaluate_benchmark_1_correctness.py \
# --pred_path results/answer-video-generic-${SAVE_NAME}.jsonl \
# --output_dir results/correctness_${SAVE_NAME} \
# --output_json results/review-video-correctness-${SAVE_NAME}.jsonl \
# --api_key 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 \
# --num_tasks 1