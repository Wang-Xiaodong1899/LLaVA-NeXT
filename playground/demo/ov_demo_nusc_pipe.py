import os
import sys

sys.path.append("/root/LLaVA-NeXT/")

import json

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

from einops import rearrange, repeat
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import os
import re
import torch
import torch.nn as nn
from tqdm import tqdm

import sys
# sys.path.append('/mnt/cache/wangxiaodong/SVD-Sense/src')

warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "/qwen/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")

model.eval()

# data config
DATAROOT = '/root/autodl-tmp/nuscenes/all'

# Function to extract frames from video
def load_video(video_path, max_frames_num, start=0, end=None):
    if type(video_path) == list:
        frame_files = video_path
        frame_files.sort()  # Ensure the frames are sorted if they are named sequentially
        
        frame_files = frame_files[start: end]
        num_frames_to_sample = max_frames_num # previous author hard code sampling 10 frames

        total_frames = len(frame_files)

        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        # Read and store the sampled frames
        video = []
        for idx in sampled_indices:
            frame_path = frame_files[idx]
            try:
                with Image.open(frame_path) as img:
                    frame = img.convert("RGB")
                    video.append(frame)
            except IOError:
                print(f"Failed to read frame at path: {frame_path}")
        return video
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


def run_inference(video_paths, max_frames_num=8, start=0, end=None):

    # Load and process video
    # video_path = "/root/LLaVA-NeXT/scene-0061"
    video_frames = load_video(video_paths, max_frames_num, start, end)
    # print(video_frames.shape) # (16, 1024, 576, 3)
    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)

    # import pdb; pdb.set_trace()

    # Prepare conversation input
    conv_template = "qwen_1_5"

    question = f"""
    {DEFAULT_IMAGE_TOKEN}
    This is a video of a car driving from a front-view camera. Please answer the following questions based on the video content. Follow the output format below. Answers should be clear, not vague.
    Weather: (e.g., sunny, cloudy, rainy, etc.)
    Time: (e.g., daytime, nighttime, etc.)
    Road environment:
    Critical objects:
    Driving action: Select one of [Speed up, Slow down, Speed up rapidly, Slow down rapidly, Go straight slowly, Go straight at a constant speed, Turn left, Turn right, Change lane to the left, Change lane to the right, Shift slightly to the left, Shift slightly to the right, Stop, Wait], or select multiple action sequences, up to a maximum of 4 action sequences.
    Scene summary: (e.g., The ego vehicle...)
    """
    # question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."
    # print(question)
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.size for frame in video_frames]

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(text_outputs[0])
    
    return text_outputs[0]


class frameDataset():
    def __init__(self, split='train'):
        self.split = split

        self.nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)

        self.splits = create_splits_scenes()
        print(self.splits)

        # training samples
        self.samples = self.get_samples(split)
        print('Total samples: %d' % len(self.samples))
        self.samples_groups = self.group_sample_by_scene(split)
    def __len__(self):
        return len(self.samples)
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict

    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(DATAROOT, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths


def parse_paragraph(paragraph):
    data = {}

    pattern = r'^(.*?):\s*(.*)$'

    lines = paragraph.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line)
        if match:
            field = match.group(1).strip()
            value = match.group(2).strip()

            if not value:
                data[field] = ""
            else:
                data[field] = value

    return data


if __name__ == "__main__":
    
    split = "val"
    dataset = frameDataset(split=split)
    scenes = list(dataset.samples_groups.keys())
    
    infer_frame = 8
    
    scene_start = 0
    scene_end = 80
    
    answers_file = os.path.join(f"nusc_video_{split}_{infer_frame}_ov-7b_{scene_start}_{scene_end}.jsonl")
    ans_file = open(answers_file, "w")
    
    for my_scene in tqdm(scenes[scene_start: scene_end]):
        paths = dataset.get_paths_from_scene(my_scene) # real video paths for captioning
        indices = [list(range(i, i+8)) for i in range(0, len(paths)-8+1, 4)]
        for items in indices:
            choose_paths = [paths[idx] for idx in items]
            output = run_inference(choose_paths, infer_frame, 0, 8)
            sample_set = parse_paragraph(output)
            
            sample_set["scene"] = my_scene
            sample_set["start"] = items[0]
            sample_set["end"] = items[-1]
            
            ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
            ans_file.flush()
    
    ans_file.close()