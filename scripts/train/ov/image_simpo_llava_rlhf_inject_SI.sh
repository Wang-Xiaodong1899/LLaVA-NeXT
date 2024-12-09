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
export WANDB_PROJECT=llava-si-image-jf-4A100
export WANDB_NAME=llava_qwen_simpo_llava-rlhf-multi-turn-aug-sft

# gpu_ids=0
gpu_ids=0,1,2,3
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

output_dir=/volsparse3/wxd/ckpt/${WANDB_PROJECT}/${WANDB_NAME}
mkdir -p $output_dir

# DATA
# data_path=/volsparse3/wxd/data/llava-onevision-data/llava_rlhf_for_dpo_ov_inject_all_fix.jsonl
# data_path=/volsparse3/wxd/data/llava-onevision-data/llava_rlhf_for_dpo_ov_multi-turn_all.jsonl
data_path=/volsparse3/wxd/data/llava-onevision-data/llava_rlhf_for_dpo_si_debate-chosen_aug-s224-rejected-sample0.6.jsonl

# sudo chmod +x -R .
# export PYTHONPATH=.

port=19006

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# DPO Stage
PROMPT_VERSION="qwen_1_5"

#torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
torchrun --nproc_per_node=$n_gpu --master_port=$port \
    llava/train/train_dpo_avg.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path /volsparse3/wxd/models/qwen/llava-onevision-qwen2-7b-si \
    --version $PROMPT_VERSION \
    --loss_type simpo \
    --dpo_alpha 1.0 --beta 2.0 --gamma 0.5 \
    --data_path=$data_path \
    --image_folder /data/mscoco/train2014 \
    --video_folder xxx \
    --mm_tunable_parts="mm_language_model" \
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
    --image_grid_pinpoints "(1x1),(1x2),(2x1)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $WANDB_NAME \
    --output_dir $output_dir \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 4 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation flash_attention_2 \