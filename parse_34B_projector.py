import safetensors
import torch


input_file_mm = "/volsparse1/wxd/models/LLaVA-NeXT-Video-34B/model-00015-of-00015.safetensors"
input_file_dpo = "/volsparse1/wxd/models/LLaVA-NeXT-Video-34B-DPO/model-00015-of-00015.safetensors"
output_file = "/volsparse1/wxd/models/LLaVA-NeXT-Video-34B/model-00015-of-00015.safetensors"

tensors_15 = safetensors.safe_open(input_file_mm, framework="pt", device="cpu")
tensors_14 = safetensors.safe_open(input_file_dpo, framework="pt", device="cpu")


mm_projector_tensors = {key: tensors_15.get_tensor(key) for key in tensors_15.keys() if "mm_projector" in key}


combined_tensors = {key: tensors_14.get_tensor(key) for key in tensors_14.keys()}
combined_tensors.update(mm_projector_tensors)

safetensors.torch.save_file(combined_tensors, output_file)

print(f"Successfully merged 'mm_projector' tensors into {output_file}")
