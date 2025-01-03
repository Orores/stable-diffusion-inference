import unittest
from src.inference import load_model, generate_image

class TestInference(unittest.TestCase):

    def test_generate_image(self):
        # Arrange
        pipe = load_model()
        prompt = "A futuristic cityscape at sunset"

        # Act
        result = generate_image(prompt, pipe)

        # Assert
        self.assertIsNotNone(result, "The generated image should not be None.")
        self.assertTrue(hasattr(result, 'save'), "The result should have a 'save' method to ensure it's an image object.")

if __name__ == '__main__':
    unittest.main()