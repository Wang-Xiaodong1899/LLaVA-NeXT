import json

qa_path = "/volsparse1/wxd/data/llava_hound/chatgpt_qa_900k.jsonl"
id_path = "/workspace/wxd/LLaVA-NeXT/long_video_ge_64.jsonl"

video_id = []
with open(qa_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        entry = json.loads(line)
        video_id.append(entry['video'])

exist_video_id = []
with open(qa_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        entry = json.loads(line)
        exist_video_id.append(entry['folder_name'])

chosen_id = []
for id in exist_video_id:
    if id in video_id:
        chosen_id.append(id)

with open('/workspace/wxd/LLaVA-NeXT/long_video_ge_64_qa.jsonl', 'w') as f:
    json.dump(chosen_id, f)