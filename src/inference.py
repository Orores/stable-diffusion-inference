import os
from diffusers import StableDiffusionPipeline
import torch
from huggingface_hub import HfApi, Repository
from dotenv import load_dotenv

load_dotenv()

def load_model(model_id="CompVis/stable-diffusion-v1-4", cache_dir="/workspace/ai_models"):
    """Load the Stable Diffusion model."""
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir=cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipe.to(device)

def generate_image(prompt, pipe):
    """Generate an image based on the given prompt using the provided model pipeline."""
    return pipe(prompt).images[0]

def upload_to_huggingface(image_path, repo_id, commit_message="Add new image"):
    """Upload the generated image to a private Hugging Face repository."""
    token = os.getenv("HF_API_TOKEN")
    api = HfApi()

    # Check if the repository exists; if not, create it
    try:
        api.repo_info(repo_id, repo_type="dataset", token=token)
    except Exception:
        api.create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True, private=True)

    # Upload the file
    api.upload_file(
        path_or_fileobj=image_path,
        path_in_repo=os.path.basename(image_path),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=commit_message
    )

if __name__ == "__main__":
    # Load model
    pipe = load_model()

    # Define the prompt
    prompt = "A futuristic cityscape at sunset"

    # Perform inference
    image = generate_image(prompt, pipe)

    # Save the generated image
    image_path = "images/output.png"
    image.save(image_path)

    # Upload the image to Hugging Face
    repo_id = "xxthekingxx/repo_2025_2"
    upload_to_huggingface(image_path, repo_id)