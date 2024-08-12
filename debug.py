import os
import warnings
import shutil
import fire

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

def test_model(model_path = "/mnt/storage/user/wangxiaodong/LLaVA-NeXT/LLaVA-NeXT-Video-7B-DPO"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    cfg_pretrained = AutoConfig.from_pretrained(model_path)
    

if __name__ == "__main__":
    fire.Fire(test_model)