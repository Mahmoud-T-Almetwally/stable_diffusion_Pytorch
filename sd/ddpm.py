import torch
import numpy as np

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alpha = 1.0 - self.betas

        self.alpha_cumprod = torch.cumprod(self.alpha, 0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self, inference_timesteps=50):
        self.num_inference_timesteps = inference_timesteps
        step_ratio = self.num_training_steps // self.num_inference_timesteps
        timesteps = (np.arange(0, self.num_inference_timesteps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alpha_cumprod.to(original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha = (1.0 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.flatten()

        while len(sqrt_one_minus_alpha.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha) * noise
        return noisy_samples
    
    def _get_previous_timestep(self, timestep: int) -> int:
        return timestep - (self.num_training_steps // self.num_inference_timesteps)
    
    def _get_variance(self, timestep: int):
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_prev_t = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_prev_t

        variance = (1 - alpha_prod_prev_t) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_prev_t = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev_t = 1 - alpha_prod_prev_t
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = (latents - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5)
        pred_original_sample_coeff = ((alpha_prod_prev_t ** 0.5) * current_beta_t ) / beta_prod_t

        current_sample_coeff = (current_alpha_t ** 0.5) * beta_prod_prev_t / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

        return pred_prev_sample + variance

    def set_strength(self, strength=1):
        start_step = self.num_inference_timesteps - int(self.num_inference_timesteps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step