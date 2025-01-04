import unittest
from src.inferencerealistic6 import load_realistic_model, generate_realistic_image
from PIL import Image
import os
import time

class TestRealisticInference(unittest.TestCase):

    def test_generate_realistic_image(self):
        # Arrange
        pipe = load_realistic_model()
        prompt = "A realistic portrait of a lion in the wild"

        # Act
        result = generate_realistic_image(prompt, pipe)

        # Assert
        self.assertIsNotNone(result, "The generated image should not be None.")
        self.assertTrue(isinstance(result, Image.Image), "The result should be an instance of PIL.Image.Image.")

    def test_realistic_image_saving(self):
        # Arrange
        pipe = load_realistic_model()
        prompt = "A realistic portrait of a lion in the wild"
        timestamp = int(time.time())
        image_path = f"images/realistic_output_{timestamp}.png"

        # Act
        result = generate_realistic_image(prompt, pipe)
        result.save(image_path)

        # Assert
        self.assertTrue(os.path.exists(image_path), "The image file should be saved with a timestamp.")

        # Cleanup
        os.remove(image_path)

if __name__ == '__main__':
    unittest.main()