import unittest
from unittest.mock import patch, MagicMock
from src.inference_realistic6_advanced import (
    load_pipeline_from_workspace,
    configure_scheduler,
    generate_image_with_params,
    save_generated_image,
    upload_to_huggingface
)
import torch

class TestInferenceRealistic6Advanced(unittest.TestCase):

    @patch('src.inference_realistic6_advanced.StableDiffusionPipeline.from_pretrained')
    def test_load_pipeline_from_workspace(self, mock_from_pretrained):
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        
        model_path = "fake_model_path"
        cache_dir = "fake_cache_dir"
        device = "cpu"
        
        pipeline = load_pipeline_from_workspace(model_path, cache_dir, device)
        
        mock_from_pretrained.assert_called_once_with(model_path, cache_dir=cache_dir, torch_dtype=torch.float16)
        self.assertEqual(pipeline, mock_pipeline.to(device))

    @patch('src.inference_realistic6_advanced.DDIMScheduler.from_config')
    @patch('src.inference_realistic6_advanced.PNDMScheduler.from_config')
    def test_configure_scheduler(self, mock_pndm_scheduler, mock_ddim_scheduler):
        mock_pipeline = MagicMock()
        mock_scheduler_config = MagicMock()
        mock_pipeline.scheduler.config = mock_scheduler_config
        
        # Test DDIM
        configure_scheduler(mock_pipeline, "DDIM")
        mock_ddim_scheduler.assert_called_once_with(mock_scheduler_config)

        # Test PNDM
        configure_scheduler(mock_pipeline, "PNDM")
        mock_pndm_scheduler.assert_called_once_with(mock_scheduler_config)
        
        # Test invalid sampler type
        with self.assertRaises(ValueError):
            configure_scheduler(mock_pipeline, "INVALID")

    @patch('src.inference_realistic6_advanced.StableDiffusionPipeline.__call__')
    def test_generate_image_with_params(self, mock_call):
        mock_pipeline = MagicMock()
        mock_image = MagicMock()
        mock_call.return_value.images = [mock_image]
        
        prompt = "test prompt"
        inference_steps = 50
        cfg_scale = 7.5
        
        image = generate_image_with_params(mock_pipeline, prompt, inference_steps, cfg_scale)
        
        mock_call.assert_called_once_with(prompt, num_inference_steps=inference_steps, guidance_scale=cfg_scale)
        self.assertEqual(image, mock_image)

    @patch('src.inference_realistic6_advanced.Image.Image.save')
    def test_save_generated_image(self, mock_save):
        mock_image = MagicMock()
        path = "fake_path.png"
        
        save_generated_image(mock_image, path)
        
        mock_image.save.assert_called_once_with(path)

    @patch('src.inference_realistic6_advanced.HfApi')
    @patch('src.inference_realistic6_advanced.os.getenv', return_value="fake_token")
    def test_upload_to_huggingface(self, mock_getenv, mock_hfapi):
        mock_api = MagicMock()
        mock_hfapi.return_value = mock_api
        
        image_path = "fake_image_path.png"
        repo_id = "fake_repo_id"
        
        upload_to_huggingface(image_path, repo_id)
        
        mock_api.repo_info.assert_called_once_with(repo_id, repo_type="dataset", token="fake_token")
        mock_api.upload_file.assert_called_once_with(
            path_or_fileobj=image_path,
            path_in_repo="fake_image_path.png",
            repo_id=repo_id,
            repo_type="dataset",
            token="fake_token",
            commit_message="Add new image"
        )

if __name__ == '__main__':
    unittest.main()