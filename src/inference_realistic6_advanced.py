import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler
from huggingface_hub import HfApi
from dotenv import load_dotenv
from PIL import Image
from realesrgan import RealESRGANer

load_dotenv()

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

def generate_image_with_params(pipeline, prompt, inference_steps, cfg_scale, width, height):
    """Generate an image using the provided parameters."""
    try:
        if pipeline.device.type == 'cuda' and torch.cuda.is_available():
            with torch.autocast(device_type='cuda'):
                image = pipeline(prompt, num_inference_steps=inference_steps, guidance_scale=cfg_scale, width=width, height=height).images[0]
        else:
            image = pipeline(prompt, num_inference_steps=inference_steps, guidance_scale=cfg_scale, width=width, height=height).images[0]
        return image
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")

def upscale_image(image, scale_factor=2):
    """Upscale the image using a pre-trained upscaler model."""
    try:
        # Path to the Real-ESRGAN weights
        model_weights_path = '/workspace/upscalers/realesr-general-x4v3.pth'
        
        # Check if the weights file exists
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Upscaler weights not found at {model_weights_path}. Please ensure the weights file is correctly placed.")

        model = RealESRGANer(scale=scale_factor, model_path=model_weights_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        upscaled_image, _ = model.enhance(image, outscale=scale_factor)
        return upscaled_image
    except Exception as e:
        raise RuntimeError(f"Image upscaling failed: {str(e)}")

def save_generated_image(image, path):
    """Save the generated image to the specified path."""
    try:
        image.save(path)
        print(f"Image saved at {path}")
    except Exception as e:
        raise RuntimeError(f"Saving image failed: {str(e)}")

def upload_to_huggingface(image_path, repo_id, commit_message="Add new image"):
    """Upload the generated image to a private Hugging Face repository."""
    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise ValueError("Hugging Face API token is not set.")
    api = HfApi()

    try:
        # Check if the repository exists; if not, create it
        api.repo_info(repo_id, repo_type="dataset", token=token)
    except Exception:
        api.create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True, private=True)

    try:
        # Upload the file
        api.upload_file(
            path_or_fileobj=image_path,
            path_in_repo=os.path.basename(image_path),
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=commit_message
        )
    except Exception as e:
        raise RuntimeError(f"Uploading to Hugging Face failed: {str(e)}")

def main():
    # Define model parameters and device
    model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    cache_dir = "/workspace/ai_models"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load the pipeline from the workspace
    pipeline = load_pipeline_from_workspace(model_path, cache_dir, device)

    # Define advanced parameters
    sampler_type = "DDIM"
    cfg_scale = 7.5
    inference_steps = 50
    prompt = "A futuristic cityscape at sunset, high resolution, ultra-detailed"
    width, height = 512, 512  # Set desired image size

    # Configure the scheduler
    configure_scheduler(pipeline, sampler_type)

    # Generate the image
    generated_image = generate_image_with_params(pipeline, prompt, inference_steps, cfg_scale, width, height)

    # Upscale the image
    upscaled_image = upscale_image(generated_image, scale_factor=2)

    # Save the upscaled image
    image_path = "upscaled_image.png"
    save_generated_image(upscaled_image, image_path)

    # Upload the image to Hugging Face
    repo_id = "xxthekingxx/realistic_repo_2025_advanced"
    upload_to_huggingface(image_path, repo_id)

if __name__ == "__main__":
    main()