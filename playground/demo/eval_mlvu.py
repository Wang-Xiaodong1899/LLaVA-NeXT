import torch
from torchvision import transforms
import json
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset

import argparse
import datasets as hf_datasets
import json
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import MAX_IMAGE_LENGTH
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoConfig

import cv2
import base64
import openai

from PIL import Image

import shortuuid

import numpy as np


def _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, image_resolution=336, video_framerate=1, s=None, e=None):
    # speed up video decode via decord.
    video_mask = np.zeros(max_frames, dtype=np.int64)
    max_video_length = 0

    # T x 3 x H x W
    video = np.zeros((max_frames, 3, image_resolution, image_resolution), dtype=np.float64)

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        
        # print(f'all pos {len(all_pos)}')
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        patch_images = torch.stack([image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images])
        slice_len = patch_images.shape[0]
        
        # print(f'slice_len: {slice_len}')

        max_video_length = max_video_length if max_video_length > slice_len else slice_len
        if slice_len < 1:
            pass
        else:
            video[:slice_len, ...] = patch_images
            # print(f"patch_images size: {patch_images.size()}")
        # print(f'len of patch_images: {len(patch_images)}')
        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))

    video_mask[:max_video_length] = [1] * max_video_length

    return torch.from_numpy(video), video_mask




def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret

class MLVU(Dataset):
    def __init__(self, data_dir, data_list):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'data': data
                })
        
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': video_path, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }



def check_ans(pred, gt):
    flag = False

    index=gt.index("(")
    index2=gt.index(")")
    gt_option=gt[index+1:index2]

    if ")" in pred:
        index3=pred.index(")")
        pred=pred[index3-1:index3]

    if pred==gt_option:
        flag=True

    return flag

def main(args):

    data_list = {
        "count": ("4_count.json", f"/data/wangxd/huggingface/hub/mlvu", "video"),
        "ego": ("3_ego.json", f"/data/wangxd/huggingface/hub/mlvu", "video"),
        "needle": ("2_needle.json", f"/data/wangxd/huggingface/hub/mlvu", "video"),
        "order": ("5_order.json", f"/data/wangxd/huggingface/hub/mlvu", "video"),
        "plotQA": ("1_plotQA.json", f"/data/wangxd/huggingface/hub/mlvu", "video"),
        "anomaly_reco": ("6_anomaly_reco.json", f"/data/wangxd/huggingface/hub/mlvu", "video"),
        "topic_reasoning": ("7_topic_reasoning.json", f"/data/wangxd/huggingface/hub/mlvu", "video")
    }
   

    data_dir = f"/data/wangxd/huggingface/hub/mlvu/json"
    save_path = f"./test_all_choice"
    result_path=f"bench_all.json"

    dataset = MLVU(data_dir, data_list)

    '''
    load your model
    '''
    if "gpt4v" != args.model_path:
        model_name = get_model_name_from_path(args.model_path)
        if "onevision" in args.model_path or "ov" in args.model_path:
            model_name = "llava_qwen"
            print(f'***********************************')
            print(f'model_name: {model_name} !!!!!!!!!')
            print(f'***********************************')
        # Set model configuration parameters if they exist
        if args.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_newline_position"] = args.mm_newline_position
            overwrite_config["enable_video_slow"] = args.enable_video_slow
            overwrite_config["enable_video_fast"] = args.enable_video_fast
            overwrite_config["enable_video_shuffle"] = args.enable_video_shuffle
            overwrite_config["enable_tube_sample"] = args.enable_tube_sample
            overwrite_config["pretrain_mm_mlp_adapter"] = args.pretrain_mm_mlp_adapter

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            # import pdb;pdb.set_trace()
            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
                else:
                    least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

                scaling_factor = math.ceil(least_token_number/4096)
                if scaling_factor >= 2:
                    if "vicuna" in cfg_pretrained._name_or_path.lower():
                        print(float(scaling_factor))
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config) #, multimodal=True)
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    else:
        pass
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']


    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    for example in tqdm(dataset):
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        video_path=example["video"]
        question=example["question"]

        '''
        modify the conv_templates like the following tempates
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        inp = [DEFAULT_VIDEO_TOKEN] '\n' + question + "Only give the best option." # noted different models take different concatenation ways
        conv.system="Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question."
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], "Best Option: (")
        prompt=get_prompt2(conv)
        '''

        '''
        run the inference code of MLLMs
        pred=run(video_path,conv_mode,prompt,...)
        '''
        qs = f"""
            Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question.
            {question}
            Only give the best option.
            Best Option: 
        """
        try:
            if video_path is not None:  # Modified this line
                video_frames, slice_len = _get_rawvideo_dec(video_path, image_processor, max_frames=args.for_get_frames_num, image_resolution=args.image_resolution)
                video_frames = [video_frames.half().cuda()]
        
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            cur_prompt = qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            if tokenizer.pad_token_id is None:
                if "qwen" in tokenizer.name_or_path.lower():
                    print("Setting pad token to bos token for qwen model.")
                    tokenizer.pad_token_id = 151643

            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            if "gpt4v" != args.model_path:
                with torch.inference_mode():
                    if "mistral" not in cfg_pretrained._name_or_path.lower():
                        output_ids = model.generate(inputs=input_ids, images=video_frames, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1,num_beams=1,use_cache=True, stopping_criteria=[stopping_criteria])
                    else:
                        output_ids = model.generate(inputs=input_ids, images=video_frames, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1, num_beams=1, use_cache=True)
            else:
                pass
            
            outputs_1 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if "mistral" not in cfg_pretrained._name_or_path.lower():
                if outputs_1.endswith(stop_str):
                    outputs_1 = outputs_1[: -len(stop_str)]
            pred = outputs_1.strip()
        except Exception as e:
            print(f"Error processing video file '{video_path}': {e}")
    
        gt = example['answer']
        res_list.append({
            'pred': pred,
            'gt': gt,
            'question':example['question'],
            'question_type':example['task_type'],
            'video':example['video']
        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)


    with open(f"{save_path}.json", "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

    final_res = dict()
    total=0
    idx=0
    for k, v in acc_dict.items():
        idx+=1
        final_res[k] = v[0] / v[1] * 100  
        total+=final_res[k]
    final_res['Avg'] = total /idx 
    print(final_res)

    with open(result_path, "w") as f:
        json.dump(final_res, f)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--video-folder", type=str, default="/workspace/wangxd/data/Video-MME/data")
    parser.add_argument("--question-file", type=str, default="/root/highspeedstorage/data/LLaVA-NeXT/llava/eval/questions/video_qa/temporal_qa.json")
    parser.add_argument("--answers-file", type=str, default="results/answer-video-mme.json")
    parser.add_argument("--subtitle", action="store_true") # TODO
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None) 

    parser.add_argument("--enable_video_slow", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--enable_video_fast", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--enable_tube_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--enable_video_shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    
    # image_resolution
    parser.add_argument("--image_resolution", type=int, default=336)
    
    
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    main(args)
