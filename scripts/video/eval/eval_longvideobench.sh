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
RESOLUTION=$9

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi

echo $RESOLUTION
    
python3 playground/demo/eval_mlvu.py \
    --model-path $CKPT \
    --output_dir ./work_dirs/longvideobench/$SAVE_DIR \
    --output_name test \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --answers-file answer-longvideobench-${SAVE_NAME}.jsonl \
    --image_resolution $RESOLUTION


# DECORD_EOF_RETRY_MAX=40960 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_longvideobench.sh /volsparse3/wxd/models/qwen/LLaVA-Video-7B-Qwen2 qwen_1_5 16 1 bilinear grid True llava-video-f16 384