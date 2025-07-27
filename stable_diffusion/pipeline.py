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
    Enhanced SDXL/SD1.x prompt encoding with per-prompt weight detection
    
    Processes positive and negative prompts separately:
    - No weights detected → Raw CLIP tokenizer (preserves bright colors)
    - Weights detected → Custom weighted processing (ComfyUI method)
    """
    
    def _detect_weights_in_prompts(*texts):
        """Ultra-precise weight detection"""
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
                    if match.startswith('(') and ':' in match:
                        parts = match.strip('()').split(':')
                        if len(parts) == 2:
                            try:
                                weight_val = float(parts[1])
                                # Only detect as weighted if weight is NOT 1.0
                                if 0.05 <= weight_val <= 20.0 and weight_val != 1.0:
                                    return True
                            except ValueError:
                                continue
        return False
    
    # Prepare batch prompts
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
    
    if len(prompt_2) != len(prompt):
        prompt_2 = prompt_2 * len(prompt) if len(prompt_2) == 1 else prompt_2[:len(prompt)]
    
    # Separate detection for positive and negative prompts
    positive_has_weights = _detect_weights_in_prompts(prompt, prompt_2)
    negative_has_weights = _detect_weights_in_prompts(negative_prompt, negative_prompt_2)
    
    # Process positive prompts
    if positive_has_weights:
        positive_embeds, positive_pooled = encode_with_weights(
            prompt, prompt_2, tokenizer, tokenizer2, 
            text_encoder, text_encoder2, device, 
            False, None, None, clip_skip  # No CFG for positive only
        )
    else:
        positive_embeds, positive_pooled = encode_with_raw_clip(
            prompt, prompt_2, tokenizer, tokenizer2, 
            text_encoder, text_encoder2, device, 
            False, None, None, clip_skip  # No CFG for positive only
        )
    
    # Handle CFG
    if do_classifier_free_guidance:
        # Prepare negative prompts
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        
        if negative_prompt_2 is None:
            negative_prompt_2 = negative_prompt.copy()
        elif isinstance(negative_prompt_2, str):
            negative_prompt_2 = [negative_prompt_2] * batch_size
        
        # Process negative prompts
        if negative_has_weights:
            negative_embeds, negative_pooled = encode_with_weights(
                negative_prompt, negative_prompt_2, 
                tokenizer, tokenizer2, text_encoder, text_encoder2, 
                device, None, None, None, clip_skip
            )
        else:
            negative_embeds, negative_pooled = encode_with_raw_clip(
                negative_prompt, negative_prompt_2, 
                tokenizer, tokenizer2, text_encoder, text_encoder2, 
                device, None, None, None, clip_skip
            )
        
        # Combine for CFG
        final_embeds = torch.cat([negative_embeds, positive_embeds])
        
        # Handle pooled embeddings
        if positive_pooled is not None and negative_pooled is not None:
            final_pooled = torch.cat([negative_pooled, positive_pooled])
        else:
            final_pooled = None
    else:
        final_embeds = positive_embeds
        final_pooled = positive_pooled
    
    return final_embeds, final_pooled

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
    Enhanced weighted processing using ComfyUI's interpolation method
    """
    
    # Detect model type
    is_sd1x = (tokenizer is tokenizer2) and (text_encoder is text_encoder2)
    
    # Prepare batch prompts
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
    
    if len(prompt_2) != len(prompt):
        prompt_2 = prompt_2 * len(prompt) if len(prompt_2) == 1 else prompt_2[:len(prompt)]
    
    def create_empty_tokens(tokenizer, max_length=77):
        """Create empty/baseline tokens for interpolation"""
        if hasattr(tokenizer, 'pad_token_id'):
            empty_token_id = tokenizer.pad_token_id
        else:
            empty_token_id = 0  # fallback
        
        return torch.full((1, max_length), empty_token_id, dtype=torch.long, device=device)
    
    def apply_weights(embeddings, empty_embeddings, token_weights, device):
        """
        Apply weights using interpolation method:
        weighted = (original - empty) * weight + empty
        """
        # Ensure token_weights has correct shape
        if len(token_weights) != embeddings.shape[1]:
            # Pad or truncate weights to match embedding sequence length
            if len(token_weights) < embeddings.shape[1]:
                token_weights.extend([1.0] * (embeddings.shape[1] - len(token_weights)))
            else:
                token_weights = token_weights[:embeddings.shape[1]]
        
        weights_tensor = torch.tensor(token_weights, device=device, dtype=embeddings.dtype)
        weights_tensor = weights_tensor.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        
        # Interpolation formula
        weighted_embeddings = (embeddings - empty_embeddings) * weights_tensor + empty_embeddings
        
        return weighted_embeddings
    
    def process_encoder_with_weights(token_weight_pairs, text_encoder, is_clip_l=True):
        """Process a single encoder with proper weight handling"""
        batch_embeddings = []
        
        # Get max sequence length
        max_len = max(len(pairs) for pairs in token_weight_pairs)
        
        # Create empty tokens for baseline
        empty_tokens = create_empty_tokens(tokenizer if is_clip_l else tokenizer2, max_len)
        
        # Get empty embeddings
        with torch.no_grad():
            if is_clip_l:
                empty_output = text_encoder(empty_tokens, clip_skip=clip_skip, output_hidden_states=True)
                empty_embeddings = empty_output if isinstance(empty_output, torch.Tensor) else empty_output[0]
            else:
                empty_embeddings, _ = text_encoder2(empty_tokens, output_hidden_states=True)
        
        for pairs in token_weight_pairs:
            # Extract tokens and weights
            tokens = [pair[0] for pair in pairs]
            weights = [pair[1] for pair in pairs]
            
            # Pad tokens to max_len
            while len(tokens) < max_len:
                tokens.append(tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0)
                weights.append(1.0)
            
            token_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            
            # Get embeddings
            with torch.no_grad():
                if is_clip_l:
                    output = text_encoder(token_ids, clip_skip=clip_skip, output_hidden_states=True)
                    embeddings = output if isinstance(output, torch.Tensor) else output[0]
                    pooled = None
                else:
                    embeddings, pooled = text_encoder2(token_ids, output_hidden_states=True)
            
            # Apply weights 
            weighted_embeddings = apply_weights(embeddings, empty_embeddings, weights, device)
            batch_embeddings.append(weighted_embeddings)
        
        return torch.cat(batch_embeddings, dim=0), pooled
    
    def process_weighted_batch(prompts_l, prompts_g, is_negative=False):
        """Enhanced batch processing with proper weight handling"""
        
        # Tokenize all prompts
        clip_l_token_weights = []
        clip_g_token_weights = []
        
        for i in range(len(prompts_l)):
            prompt_l = prompts_l[i]
            clip_l_result = tokenizer.tokenize_with_weights(prompt_l)
            
            if is_sd1x:
                if isinstance(clip_l_result, dict) and 'l' in clip_l_result:
                    clip_l_token_weights.append(clip_l_result['l'])
                elif isinstance(clip_l_result, list):
                    clip_l_token_weights.append(clip_l_result)
                else:
                    raise ValueError(f"Unexpected tokenizer format: {type(clip_l_result)}")
            else:
                prompt_g = prompts_g[i]
                clip_g_result = tokenizer2.tokenize_with_weights(prompt_g)
                clip_l_token_weights.append(clip_l_result['l'])
                clip_g_token_weights.append(clip_g_result['g'])
        
        # Process CLIP-L
        clip_l_embeddings, _ = process_encoder_with_weights(clip_l_token_weights, text_encoder, is_clip_l=True)
        
        if is_sd1x:
            return clip_l_embeddings, None
        else:
            # Process CLIP-G
            clip_g_embeddings, pooled_output = process_encoder_with_weights(clip_g_token_weights, text_encoder2, is_clip_l=False)
            
            # Combine embeddings
            combined_embeddings = torch.cat([clip_l_embeddings, clip_g_embeddings], dim=-1)
            return combined_embeddings, pooled_output
    
    # Process positive prompts
    if is_sd1x:
        prompt_embeds, _ = process_weighted_batch(prompt, prompt_2, is_negative=False)
        pooled_prompt_embeds = None
    else:
        prompt_embeds, pooled_prompt_embeds = process_weighted_batch(prompt, prompt_2, is_negative=False)
    
    # Handle Classifier-Free Guidance
    if do_classifier_free_guidance:
        # Prepare negative prompts
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        
        if negative_prompt_2 is None:
            negative_prompt_2 = negative_prompt.copy()
        elif isinstance(negative_prompt_2, str):
            negative_prompt_2 = [negative_prompt_2] * batch_size
        
        # Process negative prompts
        negative_prompt_embeds, negative_pooled = process_weighted_batch(
            negative_prompt, negative_prompt_2, is_negative=True
        )
        
        # Concatenate for CFG
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        if not is_sd1x:
            pooled_prompt_embeds = torch.cat([negative_pooled, pooled_prompt_embeds])
    
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

