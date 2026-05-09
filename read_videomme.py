import datasets as hf_datasets
from tqdm import tqdm

hf_data = hf_datasets.load_dataset("parquet", data_files="test-00000-of-00001.parquet")['train']
keys = ['video_id', 'duration', 'domain', 'sub_category', 'url', 'videoID', 'question_id', 'task_type', 'question', 'options', 'answer']

save_data = []
groups = {}

duration = "short"

# target_ques_id = "004-1"

# target_ques_id = "002-1" # N1cdUjctpG8

# target_ques_id = "041-1" # yl5ZXQmrtP0

# target_ques_id = "043-1" # 9jjTGpWmc5U

# target_ques_id = "056-1" #WmVLcj-XKnM

# target_ques_id = "058-3" # D52rTzibFRc

# target_ques_id = "079-1" # WViSvPFUVd8


## new
target_ques_id = "024-2"

target_ques_id = "051-3"

# target_ques_id = "072-2"

# target_ques_id = "071-1"


# generate answer by order
for idx in tqdm(range(len(hf_data))):
    sample = hf_data[idx]
    if sample["duration"] != duration:
        continue
    
    video_num = sample["video_id"] # eg. 001
    video_name = sample["videoID"] # eg. fFjv93ACGo8
    question_id = sample["question_id"]
    
    if question_id == target_ques_id:
        # print(sample)
        print('\n')
        print(sample["videoID"], sample["sub_category"], sample["task_type"], )