import unittest
from src.inference import load_model, generate_image

class TestHyperparameters(unittest.TestCase):

    def setUp(self):
        """Load the model once for all tests to save time."""
        self.pipe = load_model()

    def test_generate_image_default_hyperparameters(self):
        """Test image generation with default hyperparameters."""
        prompt = "cat"
        image = generate_image(prompt, self.pipe)
        self.assertIsNotNone(image, "The generated image should not be None.")
        self.assertTrue(hasattr(image, 'save'), "The result should have a 'save' method to ensure it's an image object.")

    def test_generate_image_custom_hyperparameters(self):
        """Test image generation with custom hyperparameters."""
        prompt = "dog"
        num_inference_steps = 20
        guidance_scale = 10.0
        image = generate_image(prompt, self.pipe, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        self.assertIsNotNone(image, "The generated image should not be None.")
        self.assertTrue(hasattr(image, 'save'), "The result should have a 'save' method to ensure it's an image object.")

    def test_invalid_hyperparameters(self):
        """Test image generation with invalid hyperparameters to ensure errors are raised."""
        prompt = "bird"
        with self.assertRaises(ValueError):
            generate_image(prompt, self.pipe, num_inference_steps=-1)
        with self.assertRaises(ValueError):
            generate_image(prompt, self.pipe, guidance_scale=-5.0)

if __name__ == '__main__':
    unittest.main()
