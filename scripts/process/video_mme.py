import datasets as hf_datasets
import json

def write_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f)

hf_data = hf_datasets.load_dataset("parquet", data_files=f"C://Users//wangxiaodong//Downloads//test-00000-of-00001.parquet")['train']
keys = ['video_id', 'duration', 'domain', 'sub_category', 'url', 'videoID', 'question_id', 'task_type', 'question', 'options', 'answer']

short_class = []
medium_class = []
long_class = []

groups = {}

for idx in range(len(hf_data)):
    sample = hf_data[idx]
    duration = sample['duration']
    video_num = sample["video_id"]
    if video_num not in groups:
        groups[video_num] = [sample]
    else:
        groups[video_num].append(sample)
    if duration == 'medium':
        medium_class.append(sample)
    elif duration == 'short':
        short_class.append(sample)
    else:
        long_class.append(sample)
# write_json(short_class, 'video-mme-S.json')

# write_json(medium_class, 'video-mme-M.json')

# write_json(long_class, 'video-mme-L.json')