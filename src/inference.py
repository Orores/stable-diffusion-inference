from diffusers import StableDiffusionPipeline
import torch

def load_model(model_id="CompVis/stable-diffusion-v1-4", cache_dir="/workspace/ai_models"):
    """Load the Stable Diffusion model."""
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir=cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipe.to(device)

def generate_image(prompt, pipe):
    """Generate an image based on the given prompt using the provided model pipeline."""
    return pipe(prompt).images[0]

if __name__ == "__main__":
    # Load model
    pipe = load_model()

    # Define the prompt
    prompt = "A futuristic cityscape at sunset"

    # Perform inference
    image = generate_image(prompt, pipe)

    # Save the generated image
    image.save("images/output.png")
