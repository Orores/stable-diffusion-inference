import unittest
from src.inference_sd15 import load_model_sd15, generate_image_sd15
from PIL import Image
import os
import time

class TestInferenceSD15(unittest.TestCase):

    def test_generate_image_sd15(self):
        # Arrange
        pipe = load_model_sd15()
        prompt = "A photo of an astronaut riding a horse on Mars"

        # Act
        result = generate_image_sd15(prompt, pipe)

        # Assert
        self.assertIsNotNone(result, "The generated image should not be None.")
        self.assertTrue(isinstance(result, Image.Image), "The result should be an instance of PIL.Image.Image.")

    def test_image_saving_sd15(self):
        # Arrange
        pipe = load_model_sd15()
        prompt = "A photo of an astronaut riding a horse on Mars"
        timestamp = int(time.time())
        image_path = f"images/output_sd15_{timestamp}.png"

        # Act
        result = generate_image_sd15(prompt, pipe)
        result.save(image_path)

        # Assert
        self.assertTrue(os.path.exists(image_path), "The image file should be saved with a timestamp.")

        # Cleanup
        os.remove(image_path)

if __name__ == '__main__':
    unittest.main()