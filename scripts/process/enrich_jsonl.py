import json
import os

matched_data = []

jsonl_file_path = "/volsparse3/wxd/data/self-gen/llava-next-debate-video-178k/2_3_m_youtube_v0_1-2_3_m_youtube_oe-0110/llava-next-7b-f16-s2-0_16000.jsonl"

# Read the JSONL file
with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)  # Parse the JSON line into a dictionary
        matched_data.append(entry)  # Add matching entry to the list

# matched_data = sorted(matched_data, key=lambda x: x['id'])

jsonl_file_path = "/volsparse3/wxd/data/self-gen/llava-next-aug-video-178k/2_3_m_youtube_v0_1-2_3_m_youtube_oe-0110/llava-next-7b-f2-s2-0_16000.jsonl"

matched_data_1 = []
# Read the JSONL file
with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)  # Parse the JSON line into a dictionary
        matched_data_1.append(entry)  # Add matching entry to the list

# matched_data_1 = sorted(matched_data_1, key=lambda x: x['id'])

new_data = []
# for s1, s2 in zip(matched_data, matched_data_1):
#     if s1["id"] == s2["id"]:
#         # import pdb; pdb.set_trace()
#         s1["rejected"] = s2["rejected"]
#         new_data.append(s1)
for s1 in matched_data:
    for s2 in matched_data_1:
        if s1["id"] == s2["id"] and s1["prompt"]==s2["prompt"]:
            # import pdb; pdb.set_trace()
            s1["rejected"] = s2["rejected"]
            new_data.append(s1)

print(len(new_data))
with open("/volsparse3/wxd/data/self-gen/llava-next-debate-video-178k/2_3_m_youtube_v0_1-2_3_m_youtube_oe-0110/llava-next-7b-f16-s2-merge-16k.jsonl", 'w', encoding='utf-8') as output_file:
    for item in new_data:
        json.dump(item, output_file)  # Write the JSON entry to the file
        output_file.write('\n')  # Add a newline after each entry

print(f"Kept {len(new_data)} entries.")  # Print the count of kept entries