import os
import json
from PIL import Image
import io
import datasets
from tqdm import tqdm

hf_data = datasets.load_dataset("parquet", data_files="/volsparse1/wxd/data/llava-onevision-data/llavar_gpt4_20k/train-00000-of-00002.parquet")['train']

root = "/volsparse1/wxd/data/llava-onevision-data/"

image_dir = 'llavar_gpt4_20k/images'
os.makedirs(image_dir, exist_ok=True)


json_data = []


for sample in tqdm(hf_data):
    image_data = sample['image']
    image_id = sample['id']
    conversations = sample['conversations']
    data_source = sample['data_source']
    
    image_path = os.path.join(root, image_dir, f"{image_id}.jpg")
    image_data.save(image_path)
    

    json_data.append({
        'id': image_id,
        'image': image_path,
        'conversations': conversations,
        'data_source': data_source
    })

json_output_path = '/volsparse1/wxd/data/llava-onevision-data/llavar_gpt4_20k/part_1.json'
with open(json_output_path, 'w') as f:
    json.dump(json_data, f, indent=4)

print(f"saved image to : {image_dir}")
print(f"save meta data to: {json_output_path}")
