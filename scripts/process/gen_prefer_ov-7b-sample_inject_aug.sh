#!/bin/bash
ROOT_DIR="/workspace/wxd/LLaVA-NeXT"

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
    --output_dir /volsparse1/wxd/data/self-gen/inject-1107-ov/$SAVE_DIR \
    --output_name ov-7b-inject_prior_aug_f2_sample_${START}_${END} \
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


# one-vision
# CUDA_VISIBLE_DEVICES=1 bash scripts/process/gen_prefer_ov-7b-sample_inject_aug.sh /volsparse1/wxd/models/qwen/llava-onevision-qwen2-7b-ov qwen_1_5 2 1 bilinear one_token True /volsparse1/wxd/data/llava_hound/shareVideoGPTV/QA/ /volsparse1/wxd/data/llava_hound/shareVideoGPTV/filtered_video_id_random_1_1101.jsonl 0 4000 384

# CUDA_VISIBLE_DEVICES=0 bash scripts/process/gen_prefer_ov-7b-sample_inject_aug.sh /volsparse2/wxd/models/qwen/llava-onevision-qwen2-7b-ov qwen_1_5 2 1 bilinear one_token True /volsparse1/wxd/data/llava_hound/shareVideoGPTV/QA /volsparse1/wxd/data/llava_hound/shareVideoGPTV/filtered_long_video_id_1103.jsonl 0 2000 384



# CUDA_VISIBLE_DEVICES=1 bash scripts/process/gen_prefer_ov-7b-sample_inject_aug.sh /volsparse1/wxd/ckpt/llava-ov-jf-4A100/llava_qwen_simpo_inject_prior_aug-f2-8k-longvid/checkpoint-250/ qwen_1_5 2 1 bilinear one_token True /volsparse1/wxd/data/llava_hound/shareVideoGPTV/QA/ /volsparse1/wxd/data/llava_hound/shareVideoGPTV/filtered_video_id_random_1_1101.jsonl 8000 10000 384