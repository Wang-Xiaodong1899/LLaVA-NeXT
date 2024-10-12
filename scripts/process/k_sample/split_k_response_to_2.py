import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
            
data = load_jsonl(r"C:\Users\wangxiaodong\Desktop\video_ov-7b-sample-K5\llava-onevision-qwen2-7b-ov_qwen_1_5_frames_16_stride_1\ov-7b_f16_K5_0_2000.jsonl")

k0, k1 = 2, 3

output_file = f"ov-7b_f16_K5_0_2000_k{k0}_k{k1}.jsonl"

for idx, item in enumerate(data):
    responses = item.pop("chosen")
    data[idx]["chosen"] = responses[k0]
    data[idx]["rejected"] = responses[k1]

write_jsonl(output_file, data)

