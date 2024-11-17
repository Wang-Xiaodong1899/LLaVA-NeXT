import datasets as hf_datasets
from tqdm import tqdm

hf_data = hf_datasets.load_dataset("parquet", data_files="/volsparse1/wxd/data/llava-onevision-data/llavar_gpt4_20k/train-00000-of-00002.parquet")['train']

save_data = []
groups = {}

# generate answer by order
for idx in tqdm(range(len(hf_data))):
    sample = hf_data[idx]
    print(sample.keys())
    break