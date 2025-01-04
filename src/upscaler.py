import os
import torch
from PIL import Image
import numpy as np
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Mock RealESRGAN class for demonstration purposes
class RealESRGAN:
    @staticmethod
    def from_pretrained(model_weights_path):
        return RealESRGAN()

    def __call__(self, image_tensor):
        # Mock behavior of the model for testing
        return image_tensor * 2  # Example transformation

def load_image(image_path):
    """Loads an image from the specified path."""
    return Image.open(image_path)

def save_image(image, path):
    """Saves the image to the specified path."""
    image.save(path)

def upscale_image(model, image):
    """Upscales the given image using the specified Real-ESRGAN model."""
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        upscaled_tensor = model(image_tensor)
    upscaled_image_np = upscaled_tensor.squeeze().permute(1, 2, 0).numpy()
    return Image.fromarray(upscaled_image_np.astype(np.uint8))

def upload_to_huggingface(image_path, repo_id, commit_message="Add upscaled image"):
    """Upload the upscaled image to a private Hugging Face repository."""
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

def main():
    load_dotenv()
    model_weights_path = '/workspace/upscalers/realesr-general-x4v3.pth'
    model = RealESRGAN.from_pretrained(model_weights_path)

    image_paths = [
        "images/output.png",
        "images/output_image.png",
        "images/output_sd15_1736003743.png",
        "images/realistic_output_1736004283.png"
    ]

    repo_id = "xxthekingxx/images"

    for image_path in image_paths:
        image = load_image(image_path)
        upscaled_image = upscale_image(model, image)
        upscaled_image_path = f"images/upscaled_{os.path.basename(image_path)}"
        save_image(upscaled_image, upscaled_image_path)
        upload_to_huggingface(upscaled_image_path, repo_id)

if __name__ == "__main__":
    main()