import os
import json

input_file = r'C:\Users\wangxiaodong\Desktop\llava_hound_train300k\debate-hound-qa-1230\next-7b-f16-s2-qa-0_8000.jsonl'
output_file = r'C:\Users\wangxiaodong\Desktop\llava_hound_train300k\debate-hound-qa-1230\next-7b-f16-s2-qa-0_5000.jsonl'

iter = 0
# Create the output file
with open(output_file, 'w', encoding='utf-8') as outfile:
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                if iter >= 5000:
                    break
                outfile.write(line)
                iter = iter+1

print("save completed!")
