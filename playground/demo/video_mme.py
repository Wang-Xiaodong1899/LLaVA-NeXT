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
        # print(f'len of patch_images: {len(patch_images)}')
        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))

    video_mask[:max_video_length] = [1] * max_video_length

    return torch.from_numpy(video), video_mask


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
    parser.add_argument("--video-folder", type=str, default="/volsparse1/wxd/data/Video-MME/data")
    parser.add_argument("--question-file", type=str, default="/workspace/wxd/LLaVA-NeXT/llava/eval/questions/video_qa/temporal_qa.json")
    parser.add_argument("--answers-file", type=str, default="results/answer-video-mme.json")
    parser.add_argument("--duration", type=str, default="short")
    parser.add_argument("--subtitle", action="store_true") # TODO
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None) 

    parser.add_argument("--enable_video_slow", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--enable_video_fast", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--enable_tube_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--enable_video_shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    
    
    return parser.parse_args()


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    if "gpt4v" != args.model_path:
        model_name = get_model_name_from_path(args.model_path)
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
        os.makedirs(args.output_dir)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    hf_data = hf_datasets.load_dataset("parquet", data_files="/volsparse1/wxd/data/Video-MME/test-00000-of-00001.parquet")['train']
    keys = ['video_id', 'duration', 'domain', 'sub_category', 'url', 'videoID', 'question_id', 'task_type', 'question', 'options', 'answer']

    save_data = []
    groups = {}
    

    # generate answer by order
    for idx in tqdm(range(len(hf_data))):
        sample = hf_data[idx]
        if sample["duration"] != args.duration:
            continue
        
        video_num = sample["video_id"] # eg. 001
        video_name = sample["videoID"] # eg. fFjv93ACGo8
        question_id = sample["question_id"]
        question = sample["question"]
        options = sample["options"] # eg. ["A. xxx", "B. xxx", ...]
        answer = sample["answer"]
        
        
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_folder, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break
        
        qs = f"""
        Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
        {question}
        {options[0]}
        {options[1]}
        {options[2]}
        {options[3]}
        The best answer is:
        """
        
        # print(qs)
        
        # Check if the video exists
        if video_path is not None:  # Modified this line
            video_frames, slice_len = _get_rawvideo_dec(video_path, image_processor, max_frames=args.for_get_frames_num)
            video_frames = [video_frames.half().cuda()]
        
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        if True:
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
            outputs_1 = outputs_1.strip()
            
            # save answer by video_num
            current_response = {
                "question_id": sample["question_id"],
                "task_type": sample["task_type"],
                "question": sample["question"],
                "options": sample["options"],
                "answer": sample["answer"],
                "response": outputs_1, 
            }
            if video_num not in groups:
                groups[video_num] = {
                    "video_id": video_num,
                    "duration": sample["duration"],
                    "domain": sample["domain"],
                    "sub_category": sample["sub_category"],
                    "questions": [current_response]
                }
            else:
                groups[video_num]["questions"].append(current_response)
            
        # except Exception as e:
        #     print(f"Error processing video file '{video_name}': {e}")

    for key in groups.keys():
        save_data.append(groups[key])
    
    ans_file = open(answers_file, "w")
    json.dump(save_data, ans_file)
    ans_file.close()
    

if __name__ == "__main__":
    args = parse_args()
    print(f'eval frames: {args.for_get_frames_num}')
    run_inference(args)
