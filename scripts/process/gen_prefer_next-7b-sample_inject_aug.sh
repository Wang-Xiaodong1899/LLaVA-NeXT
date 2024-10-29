#!/bin/bash
ROOT_DIR="/home/user/wangxd/LLaVA-NeXT"

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
    
python scripts/process/self_generate_preference_sample_inject_aug.py \
    --model-path $CKPT \
    --video_root ${VIDEO_PATH} \
    --output_dir /home/user/wangxd/LLaVA-NeXT/data/self-gen/simpo-iter1-inject_prior_aug-sample-1029/$SAVE_DIR \
    --output_name next-7b-simpo-iter1_inject_prior_aug_f1_sample_${START}_${END} \
    --jsonl-file $JSONLFILE \
    --start $START \
    --end $END \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --normal_frames 16 \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --image_resolution $RESOLUTION \


# one-vision
# CUDA_VISIBLE_DEVICES=0 bash scripts/process/gen_prefer_next-7b-sample_inject_aug.sh /data/wangxd/ckpt/llava_vicuna_simpo_inject_8k_aug_8k/checkpoint-300 vicuna_v1 1 2 average no_token True /home/user/wangxd/LLaVA-NeXT/data/shareVideoGPTV/QA/ /home/user/wangxd/LLaVA-NeXT/data/llava_hound/filtered_video_id.jsonl 8000 12000 224
