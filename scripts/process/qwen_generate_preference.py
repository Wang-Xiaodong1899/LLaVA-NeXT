import argparse
import torch

from qwen_vl_utils import process_vision_info
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration

import json
import os
import math
from tqdm import tqdm
import torchvision.transforms as transforms
from decord import VideoReader, cpu

from transformers import AutoConfig

import cv2
import base64
import openai

from PIL import Image



hallu_prompt_list = ["Answer this question with imaginary objects that could be in the scene. Make the anwser affirmative.",
    "Enrich your answer by adding hypothetical objects or characters that could be part of the scene. Make the anwser affirmative.",
    "Answer this question with objects or people that could logically exist in the video. Make the anwser affirmative.",
    "Enrich your answer by including elements that are not there but could fit seamlessly into the background of the video. Make the anwser affirmative.",
    "Answer this question by imagining other everyday objects or activities that take place off-screen. Make the anwser affirmative.",
    "Enrich your answer by enhancing the scene with details of possible events or objects. Make the anwser affirmative.",
    "Answer this question by imagining natural elements that could actually enter the scene, such as weather or animals. Make the anwser affirmative."]



import numpy as np

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_root", help="Path to the video files.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="/mnt/bn/wxd-video-understanding/wangxd/models/Qwen2.5-VL-7B-Instruct/")
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
    parser.add_argument("--for_get_frames_num", type=int, default=2)
    parser.add_argument("--normal_frames", type=int, default=32)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add-aug", type=bool, default=True)
    parser.add_argument("--add-hallu", type=bool, default=False) 
    parser.add_argument("--jsonl-file", type=str, default="/volsparse1/wxd/data/llava_hound/chatgpt_qa_900k.jsonl")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--skip-chosen", type=bool, default=False)
    parser.add_argument("--image_resolution", type=int, default=224) 
    
    return parser.parse_args()

import random
from PIL import ImageFilter
class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def augmentation(frame, transform, state):
    torch.set_rng_state(state)
    return transform(frame)

def load_video(video_path, args):
    if os.path.isdir(video_path):
        frame_files = [os.path.join(video_path, f) for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
        frame_files.sort()  # Ensure the frames are sorted if they are named sequentially
        num_frames_to_sample = args.normal_frames # previous author hard code sampling 10 frames

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
        
        # add augmentation
        # NOTE fix image_resolution 224
        aug_tranform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_resolution, scale=(0.08, 0.3)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip()
        ])
        
        # # save original video frame
        # for (idx, v) in enumerate(video):
        #     v.save(f'{os.path.basename(video_path)}_00{idx}.jpg')
        ori_video = video
        if args.add_aug:
            state = torch.get_rng_state()
            aug_video = [augmentation(v, aug_tranform, state) for v in video]
        else:
            aug_video = video
        
        # save aug video frame
        # for (idx, v) in enumerate(video):
        #     v.save(f'{os.path.basename(video_path)}_00{idx}_aug.jpg')
        
        # NOTE fix bug
        # for_get_frames_num not work for frames dir
        total_frame_num = len(ori_video)
        sample_frame = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_frame, dtype=int)
        aug_video = [aug_video[idx] for idx in uniform_sampled_frames]
        # import pdb; pdb.set_trace()
        print(f"video len: {len(ori_video)}, aug_video len: {len(aug_video)}")
        return ori_video, aug_video
    else:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), fps)]
        # sample_fps = args.for_get_frames_num if total_frame_num > args.for_get_frames_num else total_frame_num
        if len(frame_idx) > args.for_get_frames_num or args.force_sample:
            sample_fps = args.for_get_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # Save frames as images
        # for i, frame in enumerate(spare_frames):
        #     cv2.imwrite(f'{args.output_dir}/frame_{i}.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return spare_frames


def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    print(f"********************************")
    print(f"add-aug: {args.add_aug}")
    print(f"skip-chosen: {args.skip_chosen}")
    print(f"********************************")

    # Initialize the model
    use_vllm = False
    if not use_vllm:
        if "Qwen2-VL" in args.model_path:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16, # using float16 on V100 GPUs
                attn_implementation="flash_attention_2", # comment this line if on V100 GPUs
                device_map="auto",
            )
        elif "Qwen2.5-VL" in args.model_path:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16, # using float16 on V100 GPUs
                attn_implementation="flash_attention_2", # comment this line if on V100 GPUs
                device_map="auto",
            )
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=args.model_path,
            # gpu_memory_utilization=0.9,
            dtype=torch.bfloat16,
            limit_mm_per_prompt={"image": 32, "video": 32}, # denote the max frame number
        )

        sampling_params = SamplingParams(
            temperature=0.,
            repetition_penalty=1.05,
            max_tokens=1024,
            stop_token_ids=[],
        )
    if "Qwen2-VL" in args.model_path and "Instruct" not in args.model_path:
        processor = AutoProcessor.from_pretrained("/mnt/bn/ws-candy-hl-62827-yz89lqpbo2/models/Qwen2-VL-7B-Instruct")
        print("Loading Qwen2-VL-7B-Instruct processing class...")
    else:
        processor = AutoProcessor.from_pretrained(args.model_path)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.jsonl")
    ans_file = open(answers_file, "w")

    video_root = args.video_root

    with open(args.jsonl_file, 'r', encoding='utf-8') as file:
        jsonl_data = [json.loads(line) for line in file]

    # import pdb;pdb.set_trace()
    for item in tqdm(jsonl_data[args.start:args.end]):

        sample_set = {}
        video_ = item["video"]
        sample_set['id'] = item["id"]
        
        
        # question = next(convo['value'] for convo in item['conversations'] if convo['from'] == 'human')
        # answer = next(convo['value'] for convo in item['conversations'] if convo['from'] == 'gpt')
        
        answer = item["answer"]
        
        question = item["prompt"]

        question = question.replace("<video>\n", "")

        sample_set["prompt"] = question
        sample_set["answer"] = answer
        
        sample_set["video"] = video_
        
        
        video_path = os.path.join(video_root, video_)
        
        video = None

        # video is list of Image
        video, aug_video = load_video(video_path, args)

        # import pdb; pdb.set_trace()
        if not args.skip_chosen:
            # chosen answer
            qs = question

            prefix = "Here are some hints: " + answer + "\n\n" + "Please respond based on the given hints and video content." + "\n\n"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prefix},
                        {
                            "type": "video",
                            "video": video, "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28,
                        },
                        {"type": "text", "text": qs},
                    ],
                }
            ]
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = processor(
                text=[text],
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            print(video_kwargs)

            inputs = inputs.to("cuda")

            # Inference
            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)

            print(f'video token length: {processor.decode(generated_ids[0]).count("video_pad")}')

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            outputs_1 = output_text[0]

            outputs_1 = outputs_1.strip()
            sample_set["chosen"] = outputs_1
        
        
        # rejected answer
        question = sample_set["prompt"]
        if args.add_hallu:
            hallu_prompt = random.choice(hallu_prompt_list)
            question = question + " " + hallu_prompt
        
        if args.add_aug:
            print('---------using aug video----------')
            video = aug_video
        
        qs = question
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video, "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28,
                    },
                    {"type": "text", "text": qs},
                ],
            }
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        print(video_kwargs)

        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)

        print(f'video token length: {processor.decode(generated_ids[0]).count("video_pad")}')

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs_1 = output_text[0]

        outputs_1 = outputs_1.strip()

        sample_set["rejected"] = outputs_1
        
        ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
