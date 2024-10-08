# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# # export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

lr=${1:-"5e-7"}
ROOT=$2

# export WANDB_MODE=disabled
export WANDB_PROJECT=llava-ov-jf-4A100
export WANDB_NAME=llava-ov-qwen_dpo_17k_flash-attn_f16_blinear2_gen-16k_72b_chosen_7b_reject

# gpu_ids=0
gpu_ids=0,1,2,3
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

output_dir=/volsparse1/wxd/ckpt/${WANDB_PROJECT}/${WANDB_NAME}
mkdir -p $output_dir

# DATA
data_path=${ROOT}/ov-72b-7b_chosen_rejected_0_16000.jsonl

# sudo chmod +x -R .
# export PYTHONPATH=.

port=19002

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# DPO Stage
PROMPT_VERSION="qwen_1_5"

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
torchrun --nproc_per_node=$n_gpu --master_port=$port \
    llava/train/train_dpo.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${ROOT}/qwen/llava-onevision-qwen2-7b-ov \
    --version $PROMPT_VERSION \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
    --data_path=$data_path \
    --image_folder xxx \
    --video_folder /volsparse1/wxd/data/llava_hound/shareVideoGPTV/QA \
    --freeze_mm_mlp_adapter True \
    --frames_upbound 16 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_spatial_pool_mode "bilinear" \
    --mm_newline_position "one_token" \
    --mm_resampler_type null \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $WANDB_NAME \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3584 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation flash_attention_2 \
    --image_split_resolution 384 \
    --image_crop_resolution 384