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
cd /root/autodl-tmp/
mkdir -p data
mkdir -p data/shareVideoGPTV
cd data/shareVideoGPTV
wget https://hf-mirror.com/datasets/Xiaodong/DPO_sdf_17k/resolve/main/dpo_train_data.zip
wget https://hf-mirror.com/datasets/Xiaodong/video_instruction_train_dpo/resolve/main/sft_dpo_17k.jsonl

unzip dpo_train_data.zip
```

### Pretrained model prepare
```
mkdir -p vicuna && cd vicuna

git lfs install
git clone https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B
```

### Start train
```
# cd LLaVA-NeXT
mkdir -p ckpt

# login wandb
wandb login
# b811e4deec2c0629b9b213aa594f46f8135c13df

bash scripts/train/dpo_2_flash-attn.sh

```
