import datasets
import json
import os
import pandas as pd
from PIL import Image
import re
from tqdm import tqdm
from io import BytesIO

ROOT = "/volsparse3/wxd/data/RLHF-V-Dataset/images"


hf_data = datasets.load_dataset("parquet", data_files=["/volsparse3/wxd/data/RLHF-V-Dataset/RLHF-V-Dataset.parquet"])["train"]

# new_data = []
with open("/volsparse3/wxd/data/RLHF-V-Dataset/RLHF-V-Dataset.jsonl", 'w', encoding='utf-8') as f:
    for idx, sample in tqdm(enumerate(hf_data)):
        image = sample['image']
        image_idx = sample['idx']
        text = json.loads(sample["text"])
        question = text["question"]
        chosen = text["chosen"]
        rejected = text["rejected"]
        image = Image.open(BytesIO(image["bytes"])).convert('RGB')

        image_path_long = sample["image_path"]
        image_name = image_path_long.replace("/", "_")
        image.save(os.path.join(ROOT, image_name))

        item = {
            "origin_dataset": sample["origin_dataset"],
            "origin_split": sample["origin_split"],
            "prompt": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": image_idx,
            "image_path": image_name,
        }
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()