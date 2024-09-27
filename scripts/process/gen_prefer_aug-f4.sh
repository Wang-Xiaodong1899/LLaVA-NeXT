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
VIDEO_PATH=$8
JSONLFILE=$9
START=${10}
END=${11}
FORCE_SAMPLE=${12:-False}

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi
    
python scripts/process/self_generate_preference.py \
    --model-path $CKPT \
    --video_root ${VIDEO_PATH} \
    --add-aug True \
    --skip-chosen True \
    --output_dir /volsparse1/wxd/data/self-gen/video_aug/$SAVE_DIR \
    --output_name aug_f4_${START}_${END} \
    --jsonl-file $JSONLFILE \
    --start $START \
    --end $END \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    
    
# example
# CUDA_VISIBLE_DEVICES=0 bash scripts/process/gen_prefer_aug.sh /volsparse1/wxd/ckpt/llava-next-jf-4A100/llava_dpo_17k_flash-attn/checkpoint-3000/ vicuna_v1 4 2 average no_token True /volsparse1/wxd/data/llava_hound/QA/ /volsparse1/wxd/data/llava_hound/filtered_video_id.jsonl 0 2000