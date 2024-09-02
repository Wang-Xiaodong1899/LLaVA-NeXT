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
VIDEO_PATH=$8


if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi
    
python3 playground/demo/video_aug_demo.py \
    --model-path $CKPT \
    --video_path ${VIDEO_PATH} \
    --output_dir ./work_dirs/video_aug/$SAVE_DIR \
    --output_name __CuecKyguE \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --prompt "Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes."
    # --prompt "Please provide a very short description of the video, focusing on the main subjects, their actions. Feel free to ignore details."
    # --prompt "please fill one or two words in the blank in the following sentence: A video of ___. Focus on the main subjects and actions, tell me the filled extremely short sentence."
    # --prompt "What term describes a general concept that can represent this video? Answer in one or two words."
    # 
    
    
# example
# bash scripts/video/demo/annotate.sh /home/wxd/projects/LLaVA-NeXT/vicuna/LLaVA-NeXT-Video-7B vicuna_v1 32 2 average no_token True /mnt/storage/user/wangxiaodong/data/shareVideoGPTV/dpo_train_data