import os
import json

# Define the input folder path and output file path
input_folder = '/root/private_data/LLaVA-NeXT/results/answer-msrvtt-qa'
output_file = '/root/private_data/LLaVA-NeXT/results/answer-msrvtt-qa-llave-next-ours-0-4000.jsonl'

# Create the output file
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Iterate through each file in the folder
    filenames = list(os.listdir(input_folder))
    filenames.sort()
    print(filenames)
    for filename in filenames:
        if filename.endswith('.jsonl'):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

print("Merge complete!")
