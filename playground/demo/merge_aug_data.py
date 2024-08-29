import json
import os

if __name__ == "__main__":
    ori_jsonl_file = "/volsparse2/wxd/projects/LLaVA-NeXT/work_dirs/video_aug/LLaVA-NeXT-Video-7B_vicuna_v1_frames_32_stride_2/ori_0_5000.json"
    aug_json_file = "/volsparse2/wxd/projects/LLaVA-NeXT/work_dirs/video_aug/LLaVA-NeXT-Video-7B_vicuna_v1_frames_32_stride_2/mocov3_0_5000.json"
    
    with open(ori_jsonl_file, 'r', encoding='utf-8') as file:
        ori_data = [json.loads(line) for line in file]
    
    with open(aug_json_file, 'r', encoding='utf-8') as file:
        aug_data = [json.loads(line) for line in file]
    
    print(f'ori data length: {len(ori_data)}')
    print(f'aug data length: {len(aug_data)}')
    
    save_root = "/volsparse2/wxd/projects/LLaVA-NeXT/work_dirs/video_aug/LLaVA-NeXT-Video-7B_vicuna_v1_frames_32_stride_2"
    
    ans_file = open(os.path.join(save_root, "aug_preference_0_5000.jsonl"), "w")
    for i in range(len(ori_data)):
        ori_item = ori_data[i]
        aug_item = aug_data[i]
        prompt = ori_item["Q"]
        video = ori_item["video"]
        chosen = ori_item["pred"]
        rejected = aug_item["pred"]
        sample_set = {}
        sample_set["id"] = video
        sample_set["prompt"] = prompt
        sample_set["answer"] = ""
        sample_set["video"] = video
        sample_set["chosen"] = chosen
        sample_set["rejected"] = rejected
        
        ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()