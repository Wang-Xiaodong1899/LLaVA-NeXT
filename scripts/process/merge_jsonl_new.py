import json
import os

def merge_jsonl_files(file_list, output_file):
    """
    按顺序合并多个 jsonl 文件到一个新的 jsonl 文件

    :param file_list: 文件名列表，按顺序合并
    :param output_file: 输出的合并后的文件名
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in file_list:
            with open(file_name, 'r', encoding='utf-8') as infile:
                for line in infile:
                    
                    outfile.write(line)

root = "/volsparse3/wxd/data/self-gen/debate-iter2-hound-qa-0304/checkpoint-4000_vicuna_v1_frames_1_stride_2"

file_list = ['next-7b-f16-s2-0_1000.jsonl', 'next-7b-f16-s2-1000_2000.jsonl', 'next-7b-f16-s2-2000_3000.jsonl', 'next-7b-f16-s2-3000_4000.jsonl',
             'next-7b-f16-s2-4000_5000.jsonl', 'next-7b-f16-s2-5000_6000.jsonl', 'next-7b-f16-s2-6000_7000.jsonl', 'next-7b-f16-s2-7000_8000.jsonl'
             ]

file_list = [os.path.join(root, f) for f in file_list]

output_file = 'next-7b-f16-s2-8k.jsonl'

output_file = os.path.join(root, output_file)

merge_jsonl_files(file_list, output_file)

print(f"Saved into {output_file}")