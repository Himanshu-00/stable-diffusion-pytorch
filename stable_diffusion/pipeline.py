# pipeline.py contains the main generation pipeline function and utility functions for image generation using DDPM.

import torch
import numpy as np
from tqdm import tqdm
<<<<<<< HEAD
from samplers import KEulerAncestralSampler, KEulerSampler, KLMSSampler, DDPMSampler
=======
from samplers import KEulerAncestralSampler, KEulerSampler, KLMSSampler, DDPMSampler, DPMSolverMultistepScheduler
>>>>>>> 250a526 (fix)


def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    clip_skip=0,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
    width=512,
    height=512,
<<<<<<< HEAD
=======
    use_karras = None
>>>>>>> 250a526 (fix)
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

         # Validate dimensions
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must be multiples of 8")
            
        # Calculate latent dimensions based on input size
        latents_width = width // 8
        latents_height = height // 8

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens, clip_skip=clip_skip)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens, clip_skip=clip_skip)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens, clip_skip=clip_skip)
            
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
<<<<<<< HEAD
=======
        elif sampler_name == "dpm_solver++":
            sampler = DPMSolverMultistepScheduler(use_karras_sigmas=use_karras)
            sampler.set_inference_timesteps(n_inference_steps)
>>>>>>> 250a526 (fix)
        elif sampler_name == "k_lms":
            sampler = KLMSSampler(n_inference_steps=n_inference_steps)
        elif sampler_name == "k_euler":
            sampler = KEulerSampler(n_inference_steps=n_inference_steps)
        elif sampler_name == "k_euler_ancestral":
            sampler = KEulerAncestralSampler(n_inference_steps=n_inference_steps,
                                             generator=generator)
        else:
            raise ValueError("Unknown sampler value %s. ")


        latents_shape = (1, 4, latents_height, latents_width)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((width, height))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

<<<<<<< HEAD
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)
=======
            # # (Batch_Size, 4, Latents_Height, Latents_Width)
            # Get encoder output (8 channels)
            encoder_output = encoder(input_image_tensor)
            
            # Split into mean and log variance (4 channels each)
            mean, log_variance = torch.chunk(encoder_output, 2, dim=1)
            log_variance = torch.clamp(log_variance, -30, 20)
            variance = log_variance.exp()
            stdev = variance.sqrt()
            
            # Generate noise for latent sampling
            encoder_noise = torch.randn(mean.shape, generator=generator, device=device)
            
            # Sample latents using reparameterization trick
            latents = mean + stdev * encoder_noise
            
           
>>>>>>> 250a526 (fix)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            # time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            time_t = torch.tensor([timestep], device=device, dtype=torch.float32)  # Create 1D tensor
            model_output = diffusion(model_input, context, time_t)
            # model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

