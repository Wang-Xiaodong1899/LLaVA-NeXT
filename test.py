import json
# Read the JSONL file
jsonl_file_path = '/data/llava_hound/caption/filtered_caption_video_id_1112.jsonl'  # Replace with your JSONL file path
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)  # Parse the JSON line into a dictionary
        print(entry)
        break