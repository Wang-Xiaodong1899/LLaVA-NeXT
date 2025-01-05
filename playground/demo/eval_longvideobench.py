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


def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(':')
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds

def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    
    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]
    
    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps

def insert_subtitles(subtitles):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            subtitle_text = subtitle["text"]
        else:
            subtitle_text = subtitle["line"]

        interleaved_list.append(subtitle_text)

    return interleaved_list
        
def insert_subtitles_into_frames(frames, frame_timestamps, subtitles, 
                                 starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration
                
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            
            
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        
        for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
                if frame_timestamp <= subtitle_timestamp:
                    #print("frame:", frame_timestamp)
                    # interleaved_list.append(frame)
                    cur_i += 1
                else:
                    break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break
        #
        if covering_frames:
            #print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            #print("leaving out subtitle:", start, end)
        
    # for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
    #     #print(frame_timestamp)
    #     interleaved_list.append(frame)
        
    return interleaved_list



class LongVideoBenchDataset(Dataset):
    def __init__(self,
                 data_path,
                 annotation_file,
                 max_num_frames=16,
                 insert_text=True,
                 insert_frame=True,
                ):
        super().__init__()
        self.data_path = data_path
        self.insert_text = insert_text

        with open(os.path.join(data_path, annotation_file)) as f:
            self.data = json.load(f)
        self.max_num_frames = max_num_frames
        
        
        
    def __getitem__(self, index):
        di = self.data[index]
        
        if self.max_num_frames == 0:
            ### No subtitles, no frames        
            inputs += ["Question: " + di["question"]]
            inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
            inputs += ["Answer with the option's letter from the given choices directly."]
            return {"inputs": inputs, "correct_choice": chr(ord("A")+di["correct_choice"]), "id": di["id"]}
        if self.max_num_frames == -1:
            ### All subtitles, no frames
            with open(os.path.join(self.data_path, "subtitles", di["subtitle_path"])) as f:
                subtitles = json.load(f)
            inputs = insert_subtitles(subtitles)
            inputs += ["Question: " + di["question"]]
            inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
            inputs += ["Answer with the option's letter from the given choices directly."]
            return {"inputs": inputs, "correct_choice": chr(ord("A")+di["correct_choice"]), "id": di["id"]}
            
        frames, frame_timestamps = load_video(os.path.join(self.data_path, "videos", di["video_path"]), di["duration"], max_num_frames=self.max_num_frames)

        video_path = os.path.join(self.data_path, "videos", di["video_path"])
        
            
        with open(os.path.join(self.data_path, "subtitles", di["subtitle_path"])) as f:
            subtitles = json.load(f)
        inputs = []

        # only using subtitles
        if self.insert_text:
            selected_subtitle = insert_subtitles_into_frames(frames, frame_timestamps, subtitles, di["starting_timestamp_for_subtitles"], di["duration"])
        else:
            inputs = frames

        ##### YOU MAY MODIFY THE FOLLOWING PART TO ADAPT TO YOUR MODEL #####
        inputs += ["Question: " + di["question"]]
        inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]

        inputs += ["Answer with the option's letter from the given choices directly."]
        ##### YOU MAY MODIFY THE PREVIOUS PART TO ADAPT TO YOUR MODEL #####

        ##### CORRECT CHOICE WILL BE "@" FOR TEST SET SAMPLES #####
        # frames: PIL images
        return {"subtitle": selected_subtitle, "inputs": inputs, "correct_choice": chr(ord("A")+di.get("correct_choice", -1)), "id": di["id"], "video_path": video_path, "duration": di["duration"] }
    
    def __len__(self):
        return len(self.data)
    
    def get_id(self, index):
        return self.data[index]["id"]
    

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

    data_dir = "/volsparse3/wxd/huggingface/longvideobench"
    data_list = "lvb_val.json"

    dataset = LongVideoBenchDataset(data_dir, data_list)

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
    ans_file = open(os.path.join(args.output_dir, args.answers_file), "w")
    for example in tqdm(dataset[args.start:]):
        total += 1
        video_path=example["video_path"]
        # question=example["question"]
        question=example["inputs"]
        subtitle=example["subtitle"]

        subtitle = "\t".join(subtitle)
        question = "\n".join(question)

        qs = f"""
            Carefully watch this video and pay attention to every detail. 
            This video's subtitles are listed below:
            {subtitle}
            Based on your observations, select the best option that accurately addresses the question.
            {question}
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
    
        gt = example['correct_choice']
        sample_set = {
            'id': example['id'],
            'pred': pred,
            'gt': gt,
            'question':example['inputs'],
            'video_path':example['video_path'],
            'duration': example['duration']
        }

        if gt in pred:
            correct += 1

        ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
        ans_file.flush()

    print(f"average acc: {correct/total}")


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
    parser.add_argument("--video-folder", type=str, default="/workspace/wxd/LLaVA-NeXT/data/Video-MME/data")
    parser.add_argument("--question-file", type=str, default="/workspace/wxd/LLaVA-NeXT/llava/eval/questions/video_qa/temporal_qa.json")
    parser.add_argument("--answers-file", type=str, default="results/answer-video-mme.json")
    parser.add_argument("--subtitle", action="store_true") # TODO
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None) 

    parser.add_argument("--enable_video_slow", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--enable_video_fast", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--enable_tube_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--enable_video_shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    
    # image_resolution
    parser.add_argument("--image_resolution", type=int, default=336)

    parser.add_argument("--start", type=int, default=0)
    
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    main(args)
