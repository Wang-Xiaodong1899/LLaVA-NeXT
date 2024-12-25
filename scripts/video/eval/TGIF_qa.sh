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

python playground/demo/video_qa.py \
    --model-path $CKPT \
    --output_name "videoqa" \
    --output_dir "videoqa" \
    --question-file llava/eval/questions/video_qa/tgif_qa.json \
    --video-folder /data/VideoQA/TGIF_Zero_Shot_QA/mp4 \
    --answers-list llava/eval/questions/video_qa/tgif_a_list.json \
    --answers-file results/answer-tgif-qa-${SAVE_NAME}.jsonl \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \

# bash scripts/video/eval/TGIF_qa.sh /volsparse3/wxd/ckpt/llava-next-jf-4A100/vicuna/llava_simpo_17k_debate-hound-ours-plus-caption vicuna_v1 16 2 average no_token True llave-next-video 336