from diffusers import StableDiffusionPipeline
import torch

# First-time model download and caching
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/workspace/ai_models")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
