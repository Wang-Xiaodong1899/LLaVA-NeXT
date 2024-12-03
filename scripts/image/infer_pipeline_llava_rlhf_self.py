import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")

import os
import json
from PIL import Image
import io
import datasets
from tqdm import tqdm
import re
import torchvision.transforms as transforms
import fire
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




# *************** load model *************
model_path="/volsparse3/wxd/models/qwen/llava-onevision-qwen2-7b-ov"
model_name = get_model_name_from_path(model_path)
device = "cuda:0"
device_map = "auto"
model_base=None
tokenizer, model, image_processor, max_length = load_pretrained_model(model_path=model_path, model_base=model_base, model_name=model_name, attn_implementation='flash_attention_2')

model = model.to(device)
model.eval()

print('model loaded!')
    
def inference_pipeline(start=0, end=10000, aug=1):
    # ************* read dataset *************

    with open('/volsparse3/wxd/data/llava-onevision-data/llava_rlhf_for_dpo.json', 'r') as f:
        llava_rlhf_data = json.load(f)

    COCO_ROOT = "/data/mscoco/train2014"

    new_samples = []
    
    with open(f'/volsparse3/wxd/data/llava-onevision-data/llava_rlhf_for_dpo_ov_aug_s224_p0.6_{start}_{end}.jsonl', 'w', encoding='utf-8') as f:
        for idx, sample in tqdm(enumerate(llava_rlhf_data[start: end])):
            image_path = sample["image"] # 000000XXX.jpg
            image_id = sample["id"]
            questions = [convo['value'] for convo in sample['conversations'] if convo['from'] == 'human']
            answers = [convo['value'] for convo in sample['conversations'] if convo['from'] == 'gpt']

            question = questions[-1]
            answer = answers[-1]
            
            prompt = sample["prompt"]
            gt_answer = sample["answer"]
            
            question = question.replace('<image>', '').replace('\n', '')
            
            prompt = prompt.replace('<image>', '').replace('\n', '')
            qs = prompt
            prefix = "Here are some hints: " + gt_answer + "\n" + "Please respond based on the given hints and image content." + "\n"
            
            image = Image.open(os.path.join(COCO_ROOT, image_path))
            
            aug_tranform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 0.3)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
                transforms.RandomHorizontalFlip()
            ])
            
            aug_image = aug_tranform(image)
            
            if aug:
                image = aug_image
            
            # import pdb; pdb.set_trace()

            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
            # print("Image tensor: ", image_tensor[0].shape)
            conv_template = "qwen_1_5"
            question = DEFAULT_IMAGE_TOKEN + f"\n{qs}" 
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            # print("PROMPT: ", prompt_question)
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            image_sizes = [image.size]
            # print(image_sizes)

            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True, 
                max_new_tokens=1024,
                temperature=2.0,
                top_p=0.6,
                use_cache=True,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            # print(text_outputs[0])

            
            json_line = {
                'id': sample['id'],
                'image': sample['image'],
                'prompt': prompt,
                'answer': gt_answer,
                "rejected": text_outputs[0]
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
            f.flush()

    print(f'inference {start} to {end} done!')

if __name__ == "__main__":
    fire.Fire(inference_pipeline)