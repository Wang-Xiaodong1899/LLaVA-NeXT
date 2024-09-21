#!/bin/bash

CKPT=$1
SAVE_NAME=$2
FRAMES=$3

#eval_frame: 16 (align with finetuning)

bash scripts/video/eval/video_mme.sh $CKPT vicuna_v1 $FRAMES 2 average no_token True $SAVE_NAME
