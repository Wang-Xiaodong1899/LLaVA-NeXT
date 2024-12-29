#!/bin/bash
ROOT_DIR="/root/private_data/LLaVA-NeXT"

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
START=$9
END=${10}

python playground/demo/video_qa.py \
    --model-path $CKPT \
    --output_name "videoqa" \
    --output_dir "videoqa" \
    --question-file llava/eval/questions/video_qa/msrvtt_qa.json \
    --video-folder /root/private_data/data/VideoQA/MSRVTT_Zero_Shot_QA/videos/all \
    --answers-list llava/eval/questions/video_qa/msrvtt_a_list.json \
    --answers-file results/answer-msrvtt-qa/answer-msrvtt-qa-${SAVE_NAME}-${START}-${END}.jsonl \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --start ${START} \
    --end ${END} \

# CUDA_VISIBLE_DEVICES=3 bash scripts/video/eval/MSRVTT_qa.sh /root/private_data/ckpt/llava-next-8GPU/llava_simpo_8k_debate-hound-ours vicuna_v1 16 2 average no_token True llave-next-ours 3000 4000