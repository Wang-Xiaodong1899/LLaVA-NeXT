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
export WANDB_PROJECT=llava-next-jf-4A100
export WANDB_NAME=llava_ipo_17k_flash-attn_DPO_iter2-ov-guide

# gpu_ids=0
gpu_ids=0,1,2,3
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

output_dir=/volsparse2/wxd/ckpt/${WANDB_PROJECT}/${WANDB_NAME}
mkdir -p $output_dir

# DATA
data_path=/volsparse1/wxd/data/shareVideoGPTV/ov-72b-f32_next-7b-DPO-iter1-sample-K1_0_8000.jsonl

# sudo chmod +x -R .
# export PYTHONPATH=.

port=19002

VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

# Stage 2
PROMPT_VERSION="vicuna_v1"

#torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
torchrun --nproc_per_node=$n_gpu --master_port=$port \
    llava/train/train_dpo_avg.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path /volsparse1/wxd/ckpt/llava-next-jf-4A100/llava-next-vicuna-dpo-iter1/checkpoint-3000 \
    --version $PROMPT_VERSION \
    --dpo_alpha 1.0 --beta 0.5 --gamma 0 \
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
    --mm_spatial_pool_stride 2 \
    --mm_newline_position "no_token" \
    --mm_resampler_type "spatial_pool" \
    --mm_spatial_pool_out_channels 1024 \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $WANDB_NAME \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation flash_attention_2 \
    --image_split_resolution 224 \
    --image_crop_resolution 224