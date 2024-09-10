#!/bin/bash
ROOT=$1
CKPT=$2
SAVE_NAME=$3
FRAMES=$4

bash scripts/video/eval/temporal.sh $ROOT $CKPT $SAVE_NAME $FRAMES >> ${SAVE_NAME}_temp.txt
bash scripts/video/eval/consistency.sh $ROOT $CKPT $SAVE_NAME $FRAMES >> ${SAVE_NAME}_consi.txt