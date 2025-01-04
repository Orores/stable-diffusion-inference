import unittest
from src.inference import load_model, generate_image
from PIL import Image
import os
import time

class TestInference(unittest.TestCase):

    def test_generate_image(self):
        # Arrange
        pipe = load_model()
        prompt = "A futuristic cityscape at sunset"

        # Act
        result = generate_image(prompt, pipe)

        # Assert
        self.assertIsNotNone(result, "The generated image should not be None.")
        self.assertTrue(isinstance(result, Image.Image), "The result should be an instance of PIL.Image.Image.")

    def test_image_saving(self):
        # Arrange
        pipe = load_model()
        prompt = "A futuristic cityscape at sunset"
        timestamp = int(time.time())
        image_path = f"images/output_{timestamp}.png"

        # Act
        result = generate_image(prompt, pipe)
        result.save(image_path)

        # Assert
        self.assertTrue(os.path.exists(image_path), "The image file should be saved with a timestamp.")

        # Cleanup
        os.remove(image_path)

if __name__ == '__main__':
    unittest.main()