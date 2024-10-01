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
RESOLUTION=${12}
FORCE_SAMPLE=${13:-False}

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi
    
python scripts/process/self_generate_preference.py \
    --model-path $CKPT \
    --video_root ${VIDEO_PATH} \
    --add-aug True \
    --skip-chosen False \
    --output_dir /volsparse1/wxd/data/self-gen/video_aug/$SAVE_DIR \
    --output_name ov-7b-aug_f4_${START}_${END} \
    --jsonl-file $JSONLFILE \
    --start $START \
    --end $END \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --normal_frames 32 \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --image_resolution $RESOLUTION \
    

# example
# CUDA_VISIBLE_DEVICES=0 bash scripts/process/gen_prefer_aug-f4.sh /volsparse1/wxd/ckpt/llava-next-jf-4A100/llava_dpo_17k_flash-attn/checkpoint-3000/ vicuna_v1 4 2 average no_token True /volsparse1/wxd/data/llava_hound/shareVideoGPTV/QA/ /volsparse1/wxd/data/llava_hound/filtered_video_id.jsonl 0 2000

# one-vision
# CUDA_VISIBLE_DEVICES=0 bash scripts/process/gen_prefer_aug-f4.sh /workspace/wxd/LLaVA-NeXT/qwen/llava-onevision-qwen2-7b-ov qwen_1_5 4 1 bilinear one_token True /volsparse1/wxd/data/llava_hound/shareVideoGPTV/QA/ /volsparse1/wxd/data/llava_hound/filtered_video_id.jsonl 0 2000 384
