# To make model compatiable with implemented architecture
import torch
from safetensors import safe_open

def load_from_standard_weights(input_file: str, device: str) -> dict[str, torch.Tensor]:
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

    # MAP: UNET, (Encoder, Decoder) & CLIP
    key_mapping = {
        "encoder": ("first_stage_model.encoder.", "encoder"),
        "decoder": ("first_stage_model.decoder.", "decoder"),
        "post_quant": ("first_stage_model.post_quant_conv.", "decoder"),
        "diffusion": ("model.diffusion_model.", "diffusion")
    }

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
                
                # Diffusion model adjustments
                elif i == "diffusion":
                    updated_key = updated_key.replace("middle_block", "middle_blocks").replace(".op.", ".")
                
                break
        
        if dest and updated_key:
            converted[dest][updated_key] = original_model[key]


    # Embeddings
    converted['clip']['embedding.token_embedding.weight'] = original_model['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight']
    converted['clip']['embedding.position_embedding'] = original_model['cond_stage_model.transformer.text_model.embeddings.position_embedding.weight']
    
    # Final layer norm
    converted['clip']['layernorm.weight'] = original_model['cond_stage_model.transformer.text_model.final_layer_norm.weight']
    converted['clip']['layernorm.bias'] = original_model['cond_stage_model.transformer.text_model.final_layer_norm.bias']

    # Process each layer dynamically
    num_layers = 12 
    for i in range(num_layers):
        # Base paths
        src_base = f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn'
        dst_base = f'layers.{i}.attention'
        
        # Attention + MLP components
        components = [
            (f'{src_base}.out_proj', f'{dst_base}.out_proj'),  # Attention
            (f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm1', f'layers.{i}.layernorm_1'),
            (f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm2', f'layers.{i}.layernorm_2'),
            (f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc1', f'layers.{i}.linear_1'),
            (f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc2', f'layers.{i}.linear_2')
        ]

        # Copy weights/biases for all components
        for src_part, dst_part in components:
            for param in ['weight', 'bias']:
                converted['clip'][f'{dst_part}.{param}'] = original_model[f'{src_part}.{param}']

            
        # Concatenate QKV projections
        for param in ['weight', 'bias']:
            qkv = [original_model[f'{src_base}.{proj}_proj.{param}'] for proj in ['q', 'k', 'v']]
            converted['clip'][f'{dst_base}.in_proj.{param}'] = torch.cat(qkv, dim=0)

    return converted
