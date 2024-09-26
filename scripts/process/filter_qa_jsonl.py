import json
import os

target_directory = '/volsparse1/wxd/data/llava_hound'
# Change the current working directory
os.chdir(target_directory)

# Define the path to the JSONL file and the QA folder
jsonl_file_path = '/volsparse1/wxd/data/llava_hound/chatgpt_qa_900k.jsonl'  # Replace with your JSONL file path
qa_folder_path = 'QA'  # Replace with your QA folder path

# Get the names of all subfolders in the QA folder
video_folders = set(os.listdir(qa_folder_path))

# List to store matching entries
matched_data = []

# Read the JSONL file
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)  # Parse the JSON line into a dictionary
        if entry['video'] in video_folders:  # Check if the video exists in the QA folder
            matched_data.append(entry)  # Add matching entry to the list

# Write the matching entries to a new JSONL file
with open('filtered_video_id.jsonl', 'w') as output_file:
    for item in matched_data:
        json.dump(item, output_file)  # Write the JSON entry to the file
        output_file.write('\n')  # Add a newline after each entry

print(f"Kept {len(matched_data)} entries.")  # Print the count of kept entries
