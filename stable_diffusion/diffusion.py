# diffusion.py contains the implementation of the TimeEmbedding, UNET_ResidualBlock, UNET_AttentionBlock, Upsample, SwitchSequential & UNET.
<<<<<<< HEAD

=======
>>>>>>> 250a526 (fix)
# modified version of the original code from the CompVis/stable-diffusion repository

import math
import torch
from torch import nn
from torch.nn import functional as F
from attention import CrossAttention
from einops import rearrange, repeat
from inspect import isfunction




def normalization(channels):
    return nn.GroupNorm(32, channels)

def conv_nd(dims, in_channels, out_channels, kernel_size, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)

def zero_module(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module

def timestep_embedding(time, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time.device)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(time, 'b -> b d', d=dim)
    return embedding

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

def exists(val):
    return val is not None 

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

      


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class UNET_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_time=1280,
        dropout=False, #Used during model training
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_time = n_time

        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            conv_nd(2, in_channels, out_channels, 3, padding=1),
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(n_time, out_channels),
        )
        
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(2, out_channels, out_channels, 3, padding=1)),
        )
        
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(2, in_channels, out_channels, 1)

    def forward(self, x, time_emb):

        h = self.in_layers(x)
        time_emb = self.emb_layers(time_emb).type(h.dtype)
        while len(time_emb.shape) < len(h.shape):
            time_emb = time_emb[..., None, None]
        h = h + time_emb
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    """
    Sequential module with support for context and time inputs
    """
    def forward(self, x, context=None, time=None):
        for module in self:
            if isinstance(module, SpatialTransformer):
                x = module(x, context)
            elif isinstance(module, UNET_ResidualBlock):
                x = module(x, time)
            else:
                x = module(x)
        return x

class UNET(nn.Module):
<<<<<<< HEAD
    def __init__(self, model_channels=320, out_channels=4, dims=2):
        super().__init__()
        self.dims = dims
        self.out_channels = out_channels
        self.model_channels = model_channels
        # self.time_embedding = TimeEmbedding(320)
=======
    def __init__(self, model_channels=320, out_channels=4, in_channels=4, dims=2):
        super().__init__()
        self.dims = dims
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.model_channels = model_channels
>>>>>>> 250a526 (fix)

  
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        # Direct ModuleLists with original parameter paths
        self.input_blocks = nn.ModuleList([
<<<<<<< HEAD
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
=======
            SwitchSequential(nn.Conv2d(in_channels, 320, kernel_size=3, padding=1)),
>>>>>>> 250a526 (fix)
            SwitchSequential(UNET_ResidualBlock(320, 320), SpatialTransformer(320, n_heads=8, d_head=40, depth=1, context_dim=768)),
            SwitchSequential(UNET_ResidualBlock(320, 320), SpatialTransformer(320, n_heads=8, d_head=40, depth=1, context_dim=768)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), SpatialTransformer(640, n_heads=8, d_head=80, depth=1, context_dim=768)),
            SwitchSequential(UNET_ResidualBlock(640, 640), SpatialTransformer(640, n_heads=8, d_head=80, depth=1, context_dim=768)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), SpatialTransformer(1280, n_heads=8, d_head=160, depth=1, context_dim=768)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), SpatialTransformer(1280, n_heads=8, d_head=160, depth=1, context_dim=768)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.middle_blocks = SwitchSequential(
          UNET_ResidualBlock(1280, 1280),
          SpatialTransformer(1280, n_heads=8, d_head=160, depth=1, context_dim=768),
          UNET_ResidualBlock(1280, 1280),
        )

        self.output_blocks = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), SpatialTransformer(1280, n_heads=8, d_head=160, depth=1, context_dim=768)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), SpatialTransformer(1280, n_heads=8, d_head=160, depth=1, context_dim=768)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), SpatialTransformer(1280, n_heads=8, d_head=160, depth=1, context_dim=768), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), SpatialTransformer(640, n_heads=8, d_head=80, depth=1, context_dim=768)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), SpatialTransformer(640, n_heads=8, d_head=80, depth=1, context_dim=768)),
            SwitchSequential(UNET_ResidualBlock(960, 640), SpatialTransformer(640, n_heads=8, d_head=80, depth=1, context_dim=768), Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), SpatialTransformer(320, n_heads=8, d_head=40, depth=1, context_dim=768)),
            SwitchSequential(UNET_ResidualBlock(640, 320), SpatialTransformer(320, n_heads=8, d_head=40, depth=1, context_dim=768)),
            SwitchSequential(UNET_ResidualBlock(640, 320), SpatialTransformer(320, n_heads=8, d_head=40, depth=1, context_dim=768)),
        ])

        # self.final = UNET_OutputLayer(320, 4)
        self.out = nn.Sequential(
            normalization(self.model_channels),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, self.model_channels, self.out_channels, 3, padding=1)),
        )

    def forward(self, latent, context, time):
        # time_emb = self.time_embedding(time)
        t_emb = timestep_embedding(time, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        x = latent
        skip_connections = []
        
        # Process input_blocks
        for encoder in self.input_blocks:
            x = encoder(x, context, emb)
            skip_connections.append(x)
        
        # Process middle_blocks
        x = self.middle_blocks(x, context, emb)
        
        # Process output_blocks
        for decoder in self.output_blocks:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = decoder(x, context, emb)
        
        return self.out(x)    
    
    