# pipeline.py - SDXL/SD1x.
import torch
import numpy as np
import re
from tqdm import tqdm
from samplers import KEulerAncestralSampler, KEulerSampler, KLMSSampler, DDPMSampler, DPMSolverMultistepScheduler
from typing import Optional, Tuple

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
    width=1024,
    height=1024,
    use_karras=None,
    prompt_2: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    original_size: Optional[Tuple[int, int]] = (1024, 1024),
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = (1024, 1024),
    dtype: torch.dtype = torch.float16,
):
    with torch.no_grad():
        # Validate inputs
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must be multiples of 8")
            
        # Initialize
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
            
        latents_width = width // 8
        latents_height = height // 8
        latents_shape = (1, 4, latents_height, latents_width)
        
        # TEXT ENCODING - Supports both SD1.x and SDXL

        # Check if this is SDXL (has second text encoder)
        is_sdxl = 'clipg' in models and 'tokenizer2' in models
        
        if is_sdxl:  # SDXL MODE
            print("Running in SDXL mode")
            
            # Prepare prompts
            prompt_2 = prompt_2 or prompt
            negative_prompt = uncond_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            
            # Prepare micro-conditioning parameters
            original_size = original_size or (height, width)
            target_size = target_size or (height, width)
            
            # Encode with dual text encoders
            prompt_embeds, pooled_embeds = encode_prompt_sdxl(
                prompt=prompt,
                prompt_2=prompt_2,
                tokenizer=models['tokenizer'],
                tokenizer2= models['tokenizer2'],
                text_encoder=models['clip'],
                text_encoder2=models['clipg'],
                device=device,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                clip_skip=clip_skip,
            )

            # Ensure embeddings are in FP16
            prompt_embeds = prompt_embeds.to(dtype=dtype)
            pooled_embeds = pooled_embeds.to(dtype=dtype)

            # The UNet attention layers expect the full 2048-dimensional context
            context = prompt_embeds
            
            # Prepare time IDs for SDXL micro-conditioning
            time_ids = get_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=dtype  # Use FP16 here
            ).to(device)
            
            # For CFG, duplicate time_ids to match batch size
            if do_cfg:
                time_ids = time_ids.repeat(2, 1)

        else:  # SD1.x MODE
            print("Running in SD1.x mode")
            
            # Use the same function for consistency
            prompt_embeds, pooled_embeds = encode_prompt_sdxl(
                prompt=prompt,
                prompt_2=prompt,  # Same prompt
                tokenizer=models['tokenizer'],
                tokenizer2=models['tokenizer'],  # Same tokenizer
                text_encoder=models['clip'],
                text_encoder2=models['clip'],  # Same encoder
                device=device,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=uncond_prompt,
                negative_prompt_2=uncond_prompt,  # Same negative prompt
                clip_skip=clip_skip,
            )
            # To match variable name in diffusion loop
            context = prompt_embeds
            
            
        # Move to idle
        if idle_device:
            models['clip'].to(idle_device)
            if is_sdxl:
                models['clipg'].to(idle_device)
        
       
        # SAMPLER SETUP
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        elif sampler_name == "dpm_solver++":
            sampler = DPMSolverMultistepScheduler(use_karras_sigmas=use_karras)
            sampler.set_inference_timesteps(n_inference_steps)
        elif sampler_name == "k_lms":
            sampler = KLMSSampler(n_inference_steps=n_inference_steps)
        elif sampler_name == "k_euler":
            sampler = KEulerSampler(n_inference_steps=n_inference_steps)
        elif sampler_name == "k_euler_ancestral":
            sampler = KEulerAncestralSampler(n_inference_steps=n_inference_steps,
                                             generator=generator)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

       
        # LATENT PREPARATION & IMG2IMG
        if input_image: 
            encoder = models['encoder']
            encoder.to(device)
            
            # Process input image
            input_image_tensor = input_image.resize((width, height))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=dtype, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Encode to latent space
            encoder_output = encoder(input_image_tensor)
            mean, log_variance = torch.chunk(encoder_output, 2, dim=1)
            log_variance = torch.clamp(log_variance, -30, 20)
            variance = log_variance.exp()
            stdev = variance.sqrt()
            encoder_noise = torch.randn(mean.shape, generator=generator, device=device)
            latents = mean + stdev * encoder_noise

            # Add noise based on strength
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            if idle_device:
                encoder.to(idle_device)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)

       
        # DENOISING LOOP
       
        diffusion = models['diffusion']
        diffusion.to(device, dtype=dtype)
        
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            model_input = latents

            # For CFG, we need to handle batching differently for SDXL
            if do_cfg and not is_sdxl:
                # For SD1.x, duplicate the input for CFG
                model_input = model_input.repeat(2, 1, 1, 1)

            time_t = torch.tensor([timestep], device=device, dtype=dtype)
            
            # Call the diffusion model
            if is_sdxl:
                # For SDXL, we need to handle the batched input properly
                # Duplicate the latent input for CFG since context is already batched
                if do_cfg:
                    model_input = model_input.repeat(2, 1, 1, 1)
                
                # Check what parameters the diffusion model expects
                forward_signature = diffusion.forward.__code__.co_varnames
                
                added_cond_kwargs = {
                    "text_embeds": pooled_embeds,  # From encode_prompt_sdxl
                    "time_ids": time_ids,  # Original [batch, 6] tensor
                }

                if 'added_cond_kwargs' in forward_signature:
                    # Alternative parameter name for additional conditioning
                    model_output = diffusion(
                        model_input, 
                        context=context,
                        time=time_t,
                        added_cond_kwargs=added_cond_kwargs
                    )
                else:
                    # Basic fallback - just context
                    print("WARNING: Using basic context-only mode. SDXL features may not work properly.")
                    model_output = diffusion(
                        model_input, 
                        context=context,
                        time=time_t
                    )
                    
                # Apply CFG for SDXL - model_output should have shape [2, 4, h, w]
                if do_cfg:
                    output_uncond, output_cond = model_output.chunk(2)
                    model_output = output_uncond + cfg_scale * (output_cond - output_uncond)
            else:
                model_output = diffusion(model_input, context, time_t)
                
                # Apply CFG for SD1.x
                if do_cfg:
                    output_uncond, output_cond = model_output.chunk(2)
                    model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)

        if idle_device:
            diffusion.to(idle_device)

       
        # DECODING
        decoder = models['decoder']
        decoder.to(device, dtype=dtype)
        images = decoder(latents)

        if idle_device:
            decoder.to(idle_device)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    

# SDXL/SD1X HELPER FUNCTIONS
def encode_prompt_sdxl(
    prompt,
    prompt_2,
    tokenizer,
    tokenizer2, 
    text_encoder,
    text_encoder2,
    device,
    do_classifier_free_guidance=True,
    negative_prompt=None,
    negative_prompt_2=None,
    clip_skip=None,
):
    """
    SDX/SD1.x prompt encoding with automatic weight detection
    
    Automatically detects weights in prompts and routes to appropriate processing:
    - No weights detected → Raw CLIP tokenizer (fast, consistent)
    - Weights detected → Custom weighted processing (full weight support)
    """
    
    def _detect_weights_in_prompts(*texts):
        """Ultra-precise weight detection to avoid false positives"""
        # More restrictive: must be in parentheses with word:number format
        weight_pattern = r'\([^:)]+:[0-9]*\.?[0-9]+\)'
        
        for text in texts:
            if not text:
                continue
            if isinstance(text, list):
                texts_to_check = text
            else:
                texts_to_check = [text]
                
            for t in texts_to_check:
                if not t:
                    continue
                matches = re.findall(weight_pattern, str(t))
                for match in matches:
                    # Additional validation - ensure it's in parentheses and reasonable
                    if match.startswith('(') and ':' in match:
                        parts = match.strip('()').split(':')
                        if len(parts) == 2:
                            try:
                                weight_val = float(parts[1])
                                # Reasonable weight range for attention weights
                                if 0.05 <= weight_val <= 20.0:
                                    return True
                            except ValueError:
                                continue
        return False
    
    # Automatic weight detection
    weights_detected = _detect_weights_in_prompts(prompt, prompt_2, negative_prompt, negative_prompt_2)
    
    if weights_detected:
        return encode_with_weights(
            prompt, prompt_2, tokenizer, tokenizer2, 
            text_encoder, text_encoder2, device, 
            do_classifier_free_guidance, negative_prompt, 
            negative_prompt_2, clip_skip
        )
    else:
        return encode_with_raw_clip(
            prompt, prompt_2, tokenizer, tokenizer2, 
            text_encoder, text_encoder2, device, 
            do_classifier_free_guidance, negative_prompt, 
            negative_prompt_2, clip_skip
        )

def encode_with_raw_clip(
    prompt, prompt_2, tokenizer, tokenizer2, 
    text_encoder, text_encoder2, device, 
    do_classifier_free_guidance, negative_prompt, 
    negative_prompt_2, clip_skip
):
    """
    Raw CLIP processing - direct tokenizer access for maximum consistency
    """
    
    # Detect if this is SD1.x (same tokenizer/encoder passed twice)
    is_sd1x = (tokenizer is tokenizer2) and (text_encoder is text_encoder2)
    
    # Prepare batch prompts
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
    
    if len(prompt_2) != len(prompt):
        prompt_2 = prompt_2 * len(prompt) if len(prompt_2) == 1 else prompt_2[:len(prompt)]
    
    def process_raw_clip_batch(prompts_l, prompts_g):
        """Process batch using raw CLIP tokenizers"""
        
        # Get raw tokenizers
        raw_tokenizer_l = tokenizer.get_raw_tokenizer()
        
        # Tokenize using raw CLIP tokenizers
        tokens_l = raw_tokenizer_l(
            prompts_l, 
            padding="max_length", 
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)
        
        
        # Process with embedders
        with torch.no_grad():
            clip_l_output = text_encoder(tokens_l, clip_skip=clip_skip, output_hidden_states=True)
        
        # Handle embedder output format
        if isinstance(clip_l_output, torch.Tensor):
            clip_l_embeddings = clip_l_output
        else:
            clip_l_embeddings = clip_l_output
        
        # # Combine embeddings
        if is_sd1x:
            # SD1.x: Return single embeddings, no pooled
            return clip_l_embeddings, None
        else:
            raw_tokenizer_g = tokenizer2.get_raw_tokenizer()
            tokens_g = raw_tokenizer_g(
                prompts_g, padding="max_length",
                  max_length=77,
                  truncation=True, 
                  return_tensors="pt",
            ).input_ids.to(device)
            
            with torch.no_grad():
                clip_g_penultimate, clip_g_pooled = text_encoder2(tokens_g, output_hidden_states=True)
            # SDXL: dual encoder logic
            combined_embeddings = torch.cat([clip_l_embeddings, clip_g_penultimate], dim=-1)
        
        return combined_embeddings, clip_g_pooled
    
    if is_sd1x:
          # Process positive prompts
          prompt_embeds, _ = process_raw_clip_batch(prompt, prompt_2)
    else:
          # Process positive prompts
          prompt_embeds, pooled_prompt_embeds = process_raw_clip_batch(prompt, prompt_2)
    
    # Handle CFG
    if do_classifier_free_guidance:
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        
        if negative_prompt_2 is None:
            negative_prompt_2 = negative_prompt.copy()
        elif isinstance(negative_prompt_2, str):
            negative_prompt_2 = [negative_prompt_2] * batch_size
        
        negative_prompt_embeds, negative_pooled = process_raw_clip_batch(
            negative_prompt, negative_prompt_2
        )
        
        if is_sd1x:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            return prompt_embeds, None
        else:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_prompt_embeds = torch.cat([negative_pooled, pooled_prompt_embeds])
    else:
        # No CFG
        if is_sd1x:
            return prompt_embeds, None  # Always return tuple
        else:
            return prompt_embeds, pooled_prompt_embeds
    
    return prompt_embeds, pooled_prompt_embeds

def encode_with_weights(
    prompt, prompt_2, tokenizer, tokenizer2, 
    text_encoder, text_encoder2, device, 
    do_classifier_free_guidance, negative_prompt, 
    negative_prompt_2, clip_skip
):
    """
    Weighted processing - your existing logic with weights
    """

    # Detect if this is SD1.x (same tokenizer/encoder passed twice)
    is_sd1x = (tokenizer is tokenizer2) and (text_encoder is text_encoder2)
    
    # Prepare batch prompts
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
    
    if len(prompt_2) != len(prompt):
        prompt_2 = prompt_2 * len(prompt) if len(prompt_2) == 1 else prompt_2[:len(prompt)]
    
    def apply_token_weights(embeddings, token_weight_pairs, device):
        """Apply weights selectively to individual tokens"""
        weights = [weight for _, weight in token_weight_pairs]

        # Create weight tensor
        weights_tensor = torch.tensor(weights, device=device, dtype=embeddings.dtype)
        
        # Apply weights per-token instead of globally
        # Shape: [batch, seq_len, embed_dim] * [batch, seq_len, 1]
        weights_tensor = weights_tensor.unsqueeze(0).unsqueeze(-1)  # [1, 77, 1]
        
        # This should give proper per-token amplification
        result = embeddings * weights_tensor
        
        return result
    
    def process_weighted_batch(prompts_l, prompts_g, is_negative=False):
        """Process batch using weighted tokenizers"""
        all_clip_l_embeddings = []
        all_clip_g_embeddings = []
        all_pooled_embeddings = []
        
        for i in range(len(prompts_l)):
            prompt_l = prompts_l[i]
            
            clip_l_result = tokenizer.tokenize_with_weights(prompt_l)

            if is_sd1x:
                # SD1.x: Direct result format
                if is_sd1x:
                    # SD1.x: Handle multiple possible formats
                    if isinstance(clip_l_result, dict) and 'l' in clip_l_result:
                        # Format: {'l': [(token, weight), ...], 'g': [...]}
                        clip_l_token_weights = clip_l_result['l']
                    elif isinstance(clip_l_result, list):
                        # Format: [(token, weight), (token, weight), ...]
                        clip_l_token_weights = clip_l_result
                    else:
                        raise ValueError(f"Unexpected SD1.x tokenizer format: {type(clip_l_result)}")
            else:
                # SDXL: Dict format with 'l' and 'g' keys
                prompt_g = prompts_g[i]
                clip_g_result = tokenizer2.tokenize_with_weights(prompt_g)
                clip_l_token_weights = clip_l_result['l']
                clip_g_token_weights = clip_g_result['g']
            
            # Convert to tensors
            clip_l_ids = torch.tensor([t[0] for t in clip_l_token_weights], 
                                     dtype=torch.long).unsqueeze(0).to(device)
            
            
            # Process with embedders
            with torch.no_grad():
                clip_l_output = text_encoder(clip_l_ids, clip_skip=clip_skip, output_hidden_states=True)
            
            # Handle embedder output format
            if isinstance(clip_l_output, torch.Tensor):
                clip_l_embeddings = clip_l_output
            else:
                clip_l_embeddings = clip_l_output
            
            # Apply weights
            weighted_clip_l = apply_token_weights(clip_l_embeddings, clip_l_token_weights, device)

            all_clip_l_embeddings.append(weighted_clip_l)
        
            if is_sd1x:
                # SD1.x: No second encoder
                all_pooled_embeddings.append(None)
            else:
                # SDXL: Process second encoder
                clip_g_ids = torch.tensor(
                    [t[0] for t in clip_g_token_weights], 
                    dtype=torch.long
                ).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    clip_g_penultimate, clip_g_pooled = text_encoder2(clip_g_ids, output_hidden_states=True)
                
                weighted_clip_g = apply_token_weights(clip_g_penultimate, clip_g_token_weights, device)
                all_clip_g_embeddings.append(weighted_clip_g)
                all_pooled_embeddings.append(clip_g_pooled)
    
        batch_clip_l = torch.cat(all_clip_l_embeddings, dim=0)

        if is_sd1x:
            #   batch_clip_l = torch.cat(all_clip_l_embeddings, dim=0)
              return batch_clip_l, None
        else:
            #   batch_clip_l = torch.cat(all_clip_l_embeddings, dim=0)
              batch_clip_g = torch.cat(all_clip_g_embeddings, dim=0)
              batch_pooled = torch.cat(all_pooled_embeddings, dim=0)
        
        combined_embeddings = torch.cat([batch_clip_l, batch_clip_g], dim=-1)
        return combined_embeddings, batch_pooled
    
    if is_sd1x:
            # Process positive prompts
            prompt_embeds, _ = process_weighted_batch(prompt, prompt_2, is_negative=False)
    else:
            # Process positive prompts
            prompt_embeds, pooled_prompt_embeds = process_weighted_batch(prompt, prompt_2, is_negative=False)
    
    # Handle CFG
    if do_classifier_free_guidance:
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        
        if negative_prompt_2 is None:
            negative_prompt_2 = negative_prompt.copy()
        elif isinstance(negative_prompt_2, str):
            negative_prompt_2 = [negative_prompt_2] * batch_size
        
        negative_prompt_embeds, negative_pooled = process_weighted_batch(
            negative_prompt, negative_prompt_2, is_negative=True
        )
        
        if is_sd1x:
             prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
             return prompt_embeds, None
        else:
             prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
             pooled_prompt_embeds = torch.cat([negative_pooled, pooled_prompt_embeds])
    else:
        # No CFG
        if is_sd1x:
            return prompt_embeds, None  # Always return tuple
        else:
            return prompt_embeds, pooled_prompt_embeds
    
    return prompt_embeds, pooled_prompt_embeds

def get_time_ids(original_size, crops_coords_top_left, target_size, dtype):
    """Create time IDs for SDXL micro-conditioning"""
    return torch.tensor([
        original_size[0], original_size[1],
        crops_coords_top_left[0], crops_coords_top_left[1],
        target_size[0], target_size[1]
    ], dtype=dtype).unsqueeze(0)

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

