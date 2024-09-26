import json
import os
import shutil
from tqdm import tqdm

# Specify the path to change to
target_directory = '/volsparse1/wxd/data/llava_hound'

# Change the current working directory
os.chdir(target_directory)

# File path and data folder
jsonl_file = 'chatgpt_qa_900k.jsonl'
data_folder = 'data'
qa_folder = 'QA'

# Create QA folder if it doesn't exist
os.makedirs(qa_folder, exist_ok=True)

# Read the JSONL file and process each line
with open(jsonl_file, 'r') as file:
    for line in tqdm(file):
        data = json.loads(line.strip())
        video_folder = data['video']

        # Check if the video folder exists
        if os.path.isdir(os.path.join(data_folder, video_folder)):
            # Move the folder to the QA folder
            shutil.move(os.path.join(data_folder, video_folder), os.path.join(qa_folder, video_folder))
            print(f"Moved folder {video_folder} to {qa_folder}")

print("Processing complete!")
