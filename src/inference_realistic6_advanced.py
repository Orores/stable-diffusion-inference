import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler

def load_pipeline_from_workspace(model_path, cache_dir, device):
    """Load the Realistic Vision model pipeline from a specified cache directory."""
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, cache_dir=cache_dir, torch_dtype=torch.float16)
    return pipeline.to(device)

def configure_scheduler(pipeline, sampler_type):
    """Configure the scheduler for the pipeline based on the sampler type."""
    if sampler_type == "DDIM":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif sampler_type == "PNDM":
        pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")

def generate_image_with_params(pipeline, prompt, inference_steps, cfg_scale):
    """Generate an image using the provided parameters."""
    if pipeline.device.type == 'cuda' and torch.cuda.is_available():
        with torch.autocast(device_type='cuda'):
            image = pipeline(prompt, num_inference_steps=inference_steps, guidance_scale=cfg_scale).images[0]
    else:
        image = pipeline(prompt, num_inference_steps=inference_steps, guidance_scale=cfg_scale).images[0]
    return image

def save_generated_image(image, path):
    """Save the generated image to the specified path."""
    image.save(path)
    print(f"Image saved at {path}")

def main():
    # Define model parameters and device
    model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    cache_dir = "/workspace/ai_models"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pipeline from the workspace
    pipeline = load_pipeline_from_workspace(model_path, cache_dir, device)

    # Define advanced parameters
    sampler_type = "DDIM"
    cfg_scale = 7.5
    inference_steps = 50
    prompt = "A futuristic cityscape at sunset, high resolution, ultra-detailed"

    # Configure the scheduler
    configure_scheduler(pipeline, sampler_type)

    # Generate and save the image
    generated_image = generate_image_with_params(pipeline, prompt, inference_steps, cfg_scale)
    save_generated_image(generated_image, "generated_image.png")

if __name__ == "__main__":
    main()