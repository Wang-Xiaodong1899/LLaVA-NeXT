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
#pretrained = "/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/llava-onevision-qwen2-0.5b-finetune_multilingual_400K"
#pretrained = "/raid/phogpt_team/chitb/checkpoint_spp/llava-onevision-qwen2-0.5b-si"
model_path = "/volsparse2/wxd/models/qwen/llava-onevision-qwen2-7b-ov"
model_name = get_model_name_from_path(model_path)
device = "cuda:0"
device_map = "auto"
model_base=None
#model_base = None
tokenizer, model, image_processor, max_length = load_pretrained_model(model_path=model_path, model_base=model_base, model_name=model_name)  # Add any other thing you want to pass in llava_model_args("???", model.device)
#model = model.cuda()
model.eval()


# url = "./scripts/image/waterview.jpg"
url = "COCO_train2014_000000106644.jpg"
image = Image.open(url).convert("RGB")
print("image processor: ", image_processor)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
print("Image tensor: ", image_tensor[0].shape)
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nWhat are the differences between a muffin and a cupcake?" 
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
print("PROMPT: ", len(image_tensor), prompt_question)
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]
print(image_sizes)
# print(model.config)

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=True,
    temperature=1.0,
    top_p=0.9,
    max_new_tokens=1024,
    use_cache=True
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])