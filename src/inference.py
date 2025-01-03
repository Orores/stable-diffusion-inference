from diffusers import StableDiffusionPipeline
import torch

# Load model from local cache
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/workspace/ai_models")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Define the prompt
prompt = "A futuristic cityscape at sunset"

# Perform inference
image = pipe(prompt).images[0]

# Save the generated image
image.save("output.png")
