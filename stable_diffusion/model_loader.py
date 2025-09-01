# model_loader.py
from clip import CLIPEmbedder, CLIPGEmbedder, SDXLTokenizer, SD1Tokenizer
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import UNET
import model_converter
import torch

def get_model_config(is_sdxl):
    """Return all model configurations based on model type"""
    
    configs = {}
    
    # UNet configuration
    if is_sdxl:
        configs['unet'] = {
            'attention_resolutions': [4, 2],
            'channel_mult': [1, 2, 4],
            'context_dim': 2048,
            'num_classes': "sequential",
            'transformer_depth': [1, 2, 10],
            'n_heads_channels': 64,
            'use_linear_in_transformer': True,
        }
    else:
        configs['unet'] = {
            'context_dim': 768,
            'use_linear_in_transformer': False,
            'n_heads': 8,
            'num_classes': None,
            'attention_resolutions': [4, 2, 1],
            'channel_mult': [1, 2, 4, 4],
            'transformer_depth': 1,
        }
    
    # SDXL-specific CLIP-G configuration
    if is_sdxl:
        configs['clipl'] = {
            'layer': 'hidden',
            'layer_idx': 10,
        }
    else:
        configs['clipl'] = {}
    
    return configs


def preload_models_from_standard_weights(model_path, device, dtype=torch.float16):
    sdxl = model_converter.detect_sdxl_checkpoint(model_path)
    state_dict = model_converter.load_from_standard_weights(model_path, device, sdxl=sdxl)
    

    config = get_model_config(sdxl)

    # Load core components
    encoder = VAE_Encoder(sdxl=sdxl).to(device, dtype=torch.bfloat16)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder(sdxl=sdxl).to(device, dtype=torch.bfloat16)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = UNET(**config['unet']).to(device, dtype=dtype)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIPEmbedder(**config['clipl']).to(device, dtype=dtype)
    clip.load_state_dict(state_dict['clip'], strict=True)

    tokenizer = SD1Tokenizer()

    # Check if SDXL components exist
    models = {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
        'tokenizer': tokenizer,
    }

    # Add SDXL components if available
    if sdxl:

        clipg = CLIPGEmbedder().to(device, dtype=dtype)
        clipg.load_state_dict(state_dict['clipg'], strict=True)
        models['clipg'] = clipg

        tokenizer2 = SDXLTokenizer()
        models['tokenizer2'] = tokenizer2

    return models




