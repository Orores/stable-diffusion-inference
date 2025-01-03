import os
from huggingface_hub import HfApi
from dotenv import load_dotenv
import unittest

load_dotenv()

class TestUploadToHuggingFace(unittest.TestCase):
    def setUp(self):
        self.api = HfApi()
        self.token = os.getenv("HF_API_TOKEN")
        self.repo_id = "xxthekingxx/test"
        self.test_image_path = "tests/test_image.png"

        # Ensure the test image exists
        with open(self.test_image_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\xdac\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x8d\x1d\x00\x00\x00\x00IEND\xaeB`\x82")

    def test_upload_image(self):
        """Test uploading an image to a Hugging Face repository."""
        # Ensure the repository exists; if not, create it
        try:
            self.api.repo_info(self.repo_id, repo_type="dataset", token=self.token)
        except Exception:
            self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", token=self.token, exist_ok=True)

        # Attempt to upload the file
        try:
            self.api.upload_file(
                path_or_fileobj=self.test_image_path,
                path_in_repo=os.path.basename(self.test_image_path),
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message="Test upload of image"
            )
        except Exception as e:
            self.fail(f"Upload failed with exception: {e}")

    def tearDown(self):
        # Optionally remove the uploaded test image from the repository or leave it for verification
        pass

if __name__ == "__main__":
    unittest.main()
