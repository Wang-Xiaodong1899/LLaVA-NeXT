#!/bin/bash

CKPT=$1
SAVE_NAME=$2
FRAMES=$3
DURATION=$4

#eval_frame: 16 (align with finetuning)

bash scripts/video/eval/video_mme.sh $CKPT vicuna_v1 $FRAMES 2 average no_token True $SAVE_NAME $DURATION

python playground\demo\eval_video_mme.py \
    --results_file results/answer-video-mme-${SAVE_NAME}.json  \
    --video_duration_type $DURATION \
    --return_categories_accuracy True \