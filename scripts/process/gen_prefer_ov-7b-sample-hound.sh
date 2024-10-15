#!/bin/bash
ROOT_DIR="/root/LLaVA-NeXT"

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
    
python scripts/process/self_generate_preference_sample_hound.py \
    --model-path $CKPT \
    --video_root ${VIDEO_PATH} \
    --output_dir /root/autodl-fs/data/self-gen/video_ov-7b-sample-K5/$SAVE_DIR \
    --output_name ov-7b_f16_K5_${START}_${END} \
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
# CUDA_VISIBLE_DEVICES=0 bash scripts/process/gen_prefer_ov-7b-sample-hound.sh /root/autodl-fs/models/llava-onevision-qwen2-7b-ov qwen_1_5 16 1 bilinear one_token True /root/autodl-tmp/data/shareVideoGPTV/dpo_train_data/ /root/autodl-tmp/data/shareVideoGPTV/sft_dpo_17k.jsonl 0 2000 384
