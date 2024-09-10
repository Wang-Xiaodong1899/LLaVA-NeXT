#!/bin/bash
ROOT_DIR=$1
CKPT=$2
CONV_MODE=$3
FRAMES=$4
POOL_STRIDE=$5
POOL_MODE=$6
NEWLINE_POSITION=$7
OVERWRITE=$8
SAVE_NAME=$9

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false



if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi
    
python3 playground/demo/video_consistency.py \
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
    --answers-file results/answer-video-consistency-${SAVE_NAME}.jsonl \
    --video-folder ${ROOT_DIR}/data/Test_Videos \
    --question-file ${ROOT_DIR}/llava/eval/questions/video_qa/consistency_qa.json \

# example
# bash scripts/video/eval/video_consistency.sh /root/LLaVA-NeXT/vicuna/LLaVA-NeXT-Video-7B vicuna_v1 32 2 average no_token True LLaVA-NeXT-Video-7B
# python llava/eval/evaluate/evaluate_benchmark_5_consistency.py \
# --pred_path results/answer-video-temporal.jsonl \
# --output_dir results/temporal \
# --output_json results/review-video-temporal-xxx.jsonl \
# --api_key [openai api key] \
# --num_tasks 1