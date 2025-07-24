# To make model compatiable with implemented architecture
import re
import torch
from safetensors import safe_open


# Text encoder conversion mappings
textenc_conversion_lst = [
    ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
    ("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
    ("ln_final.weight", "text_model.final_layer_norm.weight"),
    ("ln_final.bias", "text_model.final_layer_norm.bias"),
    ("text_projection", "text_projection.weight"),
]
textenc_conversion_map = {x[0]: x[1] for x in textenc_conversion_lst}

textenc_transformer_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    ("token_embedding.weight", "transformer.text_model.embeddings.token_embedding.weight"),
    ("positional_embedding", "transformer.text_model.embeddings.position_embedding.weight"),
]
protected = {re.escape(x[0]): x[1] for x in textenc_transformer_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))


def convert_open_clip_checkpoint(
    checkpoint,
    prefix="cond_stage_model.model.",
    has_projection=False,
):

    keys = list(checkpoint.keys())
    keys_to_ignore = []
    

    text_model_dict = {}
    context_length = 77  # CLIP context length
    
    # Generate position IDs buffer
    position_ids = torch.arange(context_length).unsqueeze(0)
    if has_projection:
        text_model_dict["transformer.text_model.embeddings.position_ids"] = position_ids
    else:
        text_model_dict["text_model.embeddings.position_ids"] = position_ids

    # Handle text projection separately
    proj_key = prefix + "text_projection"
    if proj_key in checkpoint:
        text_model_dict["text_projection.weight"] = checkpoint[proj_key].T.contiguous()

    for key in keys:
        if key in keys_to_ignore:
            continue
            
        if key.startswith(prefix):
            key_without_prefix = key[len(prefix):]

            # Skip logit_scale entirely
            if key.endswith("logit_scale"):
                continue
            
            # Remove 'transformer.' prefix if present
            if key_without_prefix.startswith("transformer."):
                key_without_prefix = key_without_prefix[len("transformer."):]
            
            # Apply conversion mapping
            if key_without_prefix in textenc_conversion_map:
                new_key = textenc_conversion_map[key_without_prefix]
                text_model_dict[new_key] = checkpoint[key]
                continue
                
            # Handle attention projections
            if key_without_prefix.endswith(".in_proj_weight"):
                base_key = key_without_prefix[:-len(".in_proj_weight")]
                # Apply pattern substitution
                base_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], base_key)
                q, k, v = checkpoint[key].chunk(3, dim=0)
                text_model_dict[f"{base_key}.q_proj.weight"] = q
                text_model_dict[f"{base_key}.k_proj.weight"] = k
                text_model_dict[f"{base_key}.v_proj.weight"] = v
                continue
                
            if key_without_prefix.endswith(".in_proj_bias"):
                base_key = key_without_prefix[:-len(".in_proj_bias")]
                # Apply pattern substitution
                base_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], base_key)
                q_b, k_b, v_b = checkpoint[key].chunk(3, dim=0)
                text_model_dict[f"{base_key}.q_proj.bias"] = q_b
                text_model_dict[f"{base_key}.k_proj.bias"] = k_b
                text_model_dict[f"{base_key}.v_proj.bias"] = v_b
                continue
                
            # Apply pattern substitution to remaining keys
            new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], key_without_prefix)
            text_model_dict[new_key] = checkpoint[key]

    # Add prefix for CLIP-G models
    if has_projection:
        prefixed_dict = {}
        for key, value in text_model_dict.items():
            # Don't prefix projection or position IDs
            if key.startswith("text_projection") or "position_ids" in key:
                prefixed_dict[key] = value
            else:
                prefixed_dict["transformer." + key] = value
        return prefixed_dict

    return text_model_dict


def detect_sdxl_checkpoint(input_file: str) -> bool:
    # Check if the file is a .safetensors file
    if input_file.endswith('.safetensors'):
        # Load tensors from safetensors file to CPU first
        original_model = {}
        with safe_open(input_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                original_model[key] = f.get_tensor(key)
    else:
        # Handle .ckpt format using torch.load
        checkpoint = torch.load(input_file, map_location="cpu", weights_only=False)
        original_model = checkpoint["state_dict"]

    if any(key.startswith("conditioner.embedders.") for key in original_model.keys()):
        return True

    return False


def load_from_standard_weights(input_file: str, device: str, sdxl=None) -> dict[str, torch.Tensor]:
    # Check if the file is a .safetensors file
    if input_file.endswith('.safetensors'):
        # Load tensors from safetensors file to CPU first
        original_model = {}
        with safe_open(input_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                original_model[key] = f.get_tensor(key)
                # Convert float64 to float32 for MPS compatibility
                if original_model[key].dtype == torch.float64:
                    original_model[key] = original_model[key].to(torch.float32)
    else:
        # Handle .ckpt format using torch.load
        checkpoint = torch.load(input_file, map_location="cpu", weights_only=False)
        original_model = checkpoint["state_dict"]

    converted = {}
    converted['diffusion'] = {}
    converted['encoder'] = {}
    converted['decoder'] = {}
    converted['clip'] = {}
    converted['clipg'] = {}

    # MAP: UNET, (Encoder, Decoder) & CLIP
    key_mapping = {
        "encoder": ("first_stage_model.encoder.", "encoder"),
        "decoder": ("first_stage_model.decoder.", "decoder"),
        "post_quant": ("first_stage_model.post_quant_conv.", "decoder"),
        "diffusion": ("model.diffusion_model.", "diffusion"),
        "clip": ("conditioner.embedders.0." if sdxl else "cond_stage_model.", "clip"),
    }

    # Process standard keys first
    for key in original_model:
        updated_key = None
        dest = None
        for i, (prefix, target) in key_mapping.items():
            if key.startswith(prefix):
                dest = target
                updated_key = key[len(prefix):]  # Strip prefix
                
                # Handle post_quant_conv (no split needed)
                if i == "post_quant":
                    updated_key = f"post_quant_conv.{updated_key}"  # Directly prepend
                break

        if dest and updated_key:
            # Convert old label_emb to new add_embedding structure
            if dest == "diffusion":
                if updated_key.startswith("label_emb.0.0."):
                    # Convert: label_emb.0.0 -> add_embedding.linear_1
                    updated_key = updated_key.replace("label_emb.0.0.", "add_embedding.linear_1.")
                elif updated_key.startswith("label_emb.0.2."):
                    # Convert: label_emb.0.2 -> add_embedding.linear_2
                    updated_key = updated_key.replace("label_emb.0.2.", "add_embedding.linear_2.")
        
        if dest and updated_key:
            converted[dest][updated_key] = original_model[key]

    

    # SDXL-
    # Determine if we're dealing with a refiner or base model
    is_refiner = any(key.startswith("conditioner.embedders.0.model.") for key in original_model.keys())
    
    # Handle second text encoder for SDXL
    if any(key.startswith("conditioner.embedders.") for key in original_model.keys()):
        prefix = "conditioner.embedders.0.model." if is_refiner else "conditioner.embedders.1.model."
        
        # Convert second text encoder
        text_encoder_2 = convert_open_clip_checkpoint(
            original_model,
            prefix=prefix,
            has_projection=True,
        )
        
        # Store converted weights
        converted['clipg'] = text_encoder_2


    return converted

