import json
import os

matched_data = []

jsonl_file_path = "aug_0_8000.jsonl"

# Read the JSONL file
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)  # Parse the JSON line into a dictionary
        matched_data.append(entry)  # Add matching entry to the list

matched_data = sorted(matched_data, key=lambda x: x['id'])

jsonl_file_path = "aug_f4_0_8000.jsonl"

matched_data_1 = []
# Read the JSONL file
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)  # Parse the JSON line into a dictionary
        matched_data_1.append(entry)  # Add matching entry to the list

matched_data_1 = sorted(matched_data_1, key=lambda x: x['id'])

new_data = []
for s1, s2 in zip(matched_data, matched_data_1):
    if s1["id"] == s2["id"]:
        import pdb; pdb.set_trace()
        s2["chosen"] = s1["chosen"]
        new_data.append(s2)

print(len(new_data))
with open('aug_f4_add_chosen_0_8000.jsonl', 'w') as output_file:
    for item in new_data:
        json.dump(item, output_file)  # Write the JSON entry to the file
        output_file.write('\n')  # Add a newline after each entry

print(f"Kept {len(new_data)} entries.")  # Print the count of kept entries