#!/bin/bash
ROOT_DIR="/root/filesystem/data/LLaVA-NeXT"

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
    
python3 scripts/process/self_generate_preference_debate.py \
    --model-path $CKPT \
    --video_root ${VIDEO_PATH} \
    --output_dir /volsparse3/wxd/data/self-gen/llava-video-debate-hound-0102/$SAVE_DIR \
    --output_name llava-video-7b-f16-s2-${START}_${END} \
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


# hound
# CUDA_VISIBLE_DEVICES=0 bash scripts/process/gen_prefer_llava-video-7b-sample_debate.sh /volsparse3/wxd/models/qwen/LLaVA-Video-7B-Qwen2 qwen_1_5 16 1 bilinear grid True /data/llava_hound/shareVideoGPTV/train_300k_qa_video /volsparse3/wxd/data/shareVideoGPTV/sft_dpo_17k.jsonl 0 2000 384

# caption
# CUDA_VISIBLE_DEVICES=3 bash scripts/process/gen_prefer_next-7b-sample_debate.sh /volsparse3/wxd/models/vicuna/LLaVA-NeXT-Video-7B vicuna_v1 1 2 average no_token True /data/llava_hound/train_300k_caption_video /data/llava_hound/video_240k_caption_15k_ge_16.jsonl 9000 12000 224

#qa
# CUDA_VISIBLE_DEVICES=3 bash scripts/process/gen_prefer_next-7b-sample_debate.sh /volsparse3/wxd/models/vicuna/LLaVA-NeXT-Video-7B vicuna_v1 1 2 average no_token True /data/llava_hound/shareVideoGPTV/train_300k_qa_video /data/llava_hound/shareVideoGPTV/chatgpt_qa_900k_ge_70.jsonl 12000 16000 224