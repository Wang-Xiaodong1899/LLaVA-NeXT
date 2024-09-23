import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import random

class CustomDataset(Dataset):
    def __init__(self, list_data_dict: List[Dict[str, torch.Tensor]], num_shards: int, probabilities: List[float]):
        self.list_data_dict = sorted(list_data_dict, key=lambda x: x['logp'])
        self.num_shards = num_shards
        self.probabilities = probabilities
        self.shards = self.create_shards()

    def create_shards(self):
        # 将下标划分为 N 个 shards
        shard_size = len(self.list_data_dict) // self.num_shards
        return [list(range(i * shard_size, (i + 1) * shard_size)) for i in range(self.num_shards)]
    
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if isinstance(index, list):
            items = [self.list_data_dict[i] for i in index]
        else:
            items = [self.list_data_dict[index]]

        result = {}
        for key in items[0]:
            if isinstance(items[0][key], str):
                result[key] = [item[key] for item in items]
            elif isinstance(items[0][key], (int, float)):
                result[key] = torch.tensor([item[key] for item in items])
            elif isinstance(items[0][key], torch.Tensor):
                result[key] = torch.stack([item[key] for item in items])

        return result
    
    def sample_shards(self, total_samples: int):
        # 根据概率从 shards 中采样下标
        sampled_indices = []
        for shard, prob in zip(self.shards, self.probabilities):
            num_samples = int(total_samples * prob)
            sampled_indices.extend(random.sample(shard, min(num_samples, len(shard))))
        return sampled_indices

class CustomSampler:
    def __init__(self, dataset: CustomDataset, batch_size: int, total_samples: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_samples = total_samples

    def __iter__(self):

        while True:
            batch = []
            sampled_data = self.dataset.sample_shards(total_samples=self.total_samples)
            random.shuffle(sampled_data)
            for item in sampled_data:
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

list_data_dict = [{'logp': random.random(), 'data': 1, 'prompt': 'yes'} for _ in range(100)]
dataset = CustomDataset(list_data_dict, num_shards=5, probabilities=[0.4, 0.4, 0.1, 0.05, 0.05])
sampler = CustomSampler(dataset, batch_size=10, total_samples=50)
data_loader = DataLoader(dataset, batch_size=None, sampler=sampler)

items = []

for idx, batch in enumerate(data_loader):
    print(idx)
    if idx >=10000:
        break

    items.append(batch['logp'].numpy())

import numpy as np
import matplotlib.pyplot as plt
items = np.array(items).reshape(-1, 1)

plt.hist(items, bins=10, range=(0, 1), edgecolor='black')
plt.show()