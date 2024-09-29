## Models & Scripts

### Installation

#### 1. **Clone this repository and navigate to the LLaVA folder:**
```bash
git clone -b ding https://github.com/Wang-Xiaodong1899/LLaVA-NeXT.git
cd LLaVA-NeXT
```

#### 2. **Install the inference package:**
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

### Data prepare
```
cd LLaVA-NeXT
mkdir -p data
mkdir -p data/shareVideoGPTV
cd data/shareVideoGPTV
wget https://hf-mirror.com/datasets/Xiaodong/DPO_sdf_17k/resolve/main/dpo_train_data.zip
wget https://hf-mirror.com/datasets/Xiaodong/video_instruction_train_dpo/resolve/main/sft_dpo_17k.jsonl

unzip dpo_train_data.zip
```

### Pretrained model prepare
```
# mkdir -p vicuna && cd vicuna

mkdir -p qwen && cd qwen

# if no git-lfs, install it by
# apt-get install git-lfs

git lfs install
# git clone https://hf-mirror.com/lmms-lab/LLaVA-NeXT-Video-7B
git clone https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov
```

### Start train
```
# install flash-attention 2
pip install flash-attn --no-build-isolation

# cd LLaVA-NeXT
mkdir -p ckpt

# login wandb
wandb login
# b811e4deec2c0629b9b213aa594f46f8135c13df

# bash scripts/train/dpo_2_flash-attn.sh 5e-7 ./

# (9.10 training)
# sync code
# git reset --hard origin/ding
# bash scripts/train/dpo_2_cf_flash-attn-fast-reg-no-rej.sh 5e-7 ./
# bash scripts/train/dpo_2_cs_flash-attn-slow-reg-no-rej.sh 5e-7 ./

# (9.29 training)
# sync code
git reset --hard origin/ding

bash scripts/train/ov/dpo_ov7b.sh 5e-7 ./

```

### Evaluation on Video-MME
```
cd LLaVA-NeXT
mkdir -p data && cd data
mkdir -p Video-MME && cd Video-MME
# metadata
wget https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main/videomme/test-00000-of-00001.parquet

# video data
# download zip
bash scripts/process/video_mme/video_mme.sh

# unzip
bash scripts/process/video_mme/unzip.sh


# start eval on 1 GPU
# short
export DECORD_EOF_RETRY_MAX=20480 && CUDA_VISIBLE_DEVICES=0 bash scripts/video/eval/eval_video_mme.sh ckpt_path 7b-ov-DPO-short 32 short True qwen_1_5 1 bilinear one_token 384
# medium
export DECORD_EOF_RETRY_MAX=20480 && CUDA_VISIBLE_DEVICES=1 bash scripts/video/eval/eval_video_mme.sh ckpt_path 7b-ov-DPO-medium 32 medium True qwen_1_5 1 bilinear one_token 384
# long
export DECORD_EOF_RETRY_MAX=20480 && CUDA_VISIBLE_DEVICES=2 bash scripts/video/eval/eval_video_mme.sh ckpt_path 7b-ov-DPO-long 32 long True qwen_1_5 1 bilinear one_token 384

```

### **~~Evaluation (only need 1 GPU)~~**
```
conda activate llava
pip install jsonlines

# cd LLaVA-NeXT
cd data
wget https://hf-mirror.com/datasets/lmms-lab/VideoChatGPT/resolve/main/videos.zip
unzip videos.zip

# temporal and consistency eval
# cd LLaVA-NeXT
# bash scripts/video/eval/temp_consi.sh ./ ./ckpt/llava-next-8-H100-1/llava_dpo_17k_flash-attn dpo_17k 16

# bash scripts/video/eval/temp_consi.sh ./ ./ckpt/llava-next-8-H100-1/llava_dpo_17k_nll-loss-chosen-fast-reg-no-rej fast-reg-no-rej 16


bash scripts/video/eval/temp_consi.sh ./ ./ckpt/llava-next-8-H100-1/llava_dpo_17k_nll-loss-chosen-slow-reg-no-rej slow-reg-no-rej 16

# It may cost 2 hours, after that, you will see two txt file: _temp.txt and _consi.txt

```
