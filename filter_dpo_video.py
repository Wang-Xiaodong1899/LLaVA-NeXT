import json
import os
from tqdm import tqdm

def filter_id():
    jsonl_file_path = '/mnt/storage/user/wangxiaodong/data/Hound-DPO/sft_dpo_17k.jsonl'
    json_file_path = '/mnt/storage/user/wangxiaodong/data/Hound-DPO/sft_dpo_17k_id.json'

    ids = []


    with open(jsonl_file_path, 'r') as jsonl_file:
        for line in jsonl_file:

            data = json.loads(line)

            if 'id' in data:
                ids.append(data['id'])


    with open(json_file_path, 'w') as json_file:
        json.dump(ids, json_file)

    print(f"All id saved in {json_file_path}")

def filter_directory():

    # 输入的 JSON 文件路径
    json_file_path = '/mnt/storage/user/wangxiaodong/data/Hound-DPO/sft_dpo_17k_id.json'  # 存储 ID 的文件
    # 需要检查的目录路径
    directory_path = '/mnt/storage/user/wangxiaodong/data/Hound-DPO/train_300k'  # 替换为你的目录路径
    # 输出的 JSON 文件路径
    filtered_json_file_path = '/mnt/storage/user/wangxiaodong/data/Hound-DPO/sft_dpo_17k_filtered_ids.json'

    # 读取 JSON 文件中的 IDs
    with open(json_file_path, 'r') as json_file:
        ids = json.load(json_file)

    # 转换 IDs 为集合以提高查找速度
    id_set = set(ids)

    # 存储匹配的 IDs
    filtered_ids = []

    # 遍历目录中的所有文件夹
    for folder_name in tqdm(os.listdir(directory_path)):
        if os.path.isdir(os.path.join(directory_path, folder_name)):
            # 如果文件夹名称在 ID 集合中，则保留
            if folder_name in id_set:
                filtered_ids.append(folder_name)

    # 将匹配的 IDs 写入新的 JSON 文件
    with open(filtered_json_file_path, 'w') as filtered_json_file:
        json.dump(filtered_ids, filtered_json_file)

    print(f"匹配的 IDs 已保存到 {filtered_json_file_path}")


def filter_item_by_id():
    jsonl_file_path = '/mnt/storage/user/wangxiaodong/data/Hound-DPO/sft_dpo_17k.jsonl'
    new_jsonl_file_path = '/mnt/storage/user/wangxiaodong/data/Hound-DPO/sft_dpo_17k_ours.json'

    ids = []
    
    data_root = '/mnt/storage/user/wangxiaodong/data/Hound-DPO/dpo_train_data'
    
    folder_ids = []
    folders = os.listdir(data_root)
    for f in folders:
        if os.path.isdir(os.path.join(data_root, f)):
            folder_ids.append(f)
    
    print(f'foloder length: {len(folder_ids)}')
    
    all_data = []

    existing_items = []
    with open(jsonl_file_path, 'r') as jsonl_file:
        for line in jsonl_file:

            data = json.loads(line)
            all_data.append(data)

            if 'video' in data:
                if data['video'] in folder_ids:
                    existing_items.append(data)
    
    print(f'all items {len(all_data)}')
    print(f'selected items {len(existing_items)}')
    
    with open(new_jsonl_file_path, 'w') as jsonl_file:
        for entry in existing_items:
            jsonl_file.write(json.dumps(entry) + '\n')

filter_item_by_id()