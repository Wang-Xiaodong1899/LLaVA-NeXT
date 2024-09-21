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
FORCE_SAMPLE=${9:-False}


if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi
    
python3 scripts/process/analysis_logp.py \
    --model-path $CKPT \
    --video_path ${VIDEO_PATH} \
    --output_dir ./work_dirs/dpo_17k_data/$SAVE_DIR \
    --output_name test \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --enable_video_slow False \
    --enable_video_fast False \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --force_sample $FORCE_SAMPLE \

# example
# bash scripts/video/demo/video_demo.sh /root/autodl-tmp/vicuna/LLaVA-NeXT-Video-7B vicuna_v1 32 2 average no_token True playground/demo/xU25MMA2N4aVtYay.mp4 True

# video case What are the men playing on the beach? 
# TODO test whether to affect the DPO model following ability
# bash scripts/video/demo/video_demo.sh /root/LLaVA-NeXT/vicuna/LLaVA-NeXT-Video-7B-DPO vicuna_v1 32 2 average no_token True playground/demo/v_oR8o_PuKS28.mp4

# 34B
# bash scripts/video/demo/video_demo.sh /volsparse1/wxd/models/LLaVA-NeXT-Video-34B-DPO/ mistral_direct 16 2 average no_token True playground/demo/v_oR8o_PuKS28.mp4