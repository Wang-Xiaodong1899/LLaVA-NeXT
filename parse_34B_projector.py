import torch
from safetensors.torch import load_file


file_path = "/volsparse1/wxd/models/LLaVA-NeXT-Video-34B/model-00015-of-00015.safetensors"
weights = load_file(file_path)

mm_projector_weights = {k: v for k, v in weights.items() if 'mm_projector' in k}


torch.save(mm_projector_weights, "/volsparse1/wxd/models/LLaVA-NeXT-Video-34B/mm_projector_weights.pt")

print("Saved mm_projector_weights.pt")
