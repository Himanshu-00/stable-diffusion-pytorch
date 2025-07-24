# diffusion.py contains the implementation of the TimeEmbedding, UNET_ResidualBlock, UNET_AttentionBlock, Upsample, Downsample, SwitchSequential & UNET.
# modified version of the original code from the official repository

import math
import torch
from torch import nn
import torch as th
from torch.nn import functional as F
from attention import CrossAttention
from einops import rearrange, repeat
from inspect import isfunction
from typing import Iterable, Optional, Union, List, Tuple, Dict
import logging




logpy = logging.getLogger(__name__)


def normalization(channels):
    return nn.GroupNorm(32, channels)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

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
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=time.dtype) / half
        ).to(device=time.device)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(time, 'b -> b d', d=dim)
    return embedding.to(dtype=time.dtype)

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
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # cross attention
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        disable_self_attn=False,
        attn_mode="softmax",
    ):
        super().__init__()

        self.attn_mode = attn_mode
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        return x
    
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
    ):
        super().__init__()
        logpy.debug(
            f"constructing {self.__class__.__name__} of depth {depth} w/ "
            f"{in_channels} channels and {n_heads} heads."
        )

        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in  

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        third_down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))
        if use_conv:
            logpy.info(f"Building a Downsample layer with {dims} dims.")
            logpy.info(
                f"  --> settings are: \n in-chn: {self.channels}, out-chn: {self.out_channels}, "
                f"kernel-size: 3, stride: {stride}, padding: {padding}"
            )
            if dims == 3:
                logpy.info(f"  --> Downsampling third axis (time): {third_down}")
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            # self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels

        return self.op(x)

class UNET_ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        kernel_size: int = 3,
        exchange_temb_dims: bool = False,
        skip_t_emb: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            logpy.info(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    self.emb_out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    
    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = th.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        third_up: bool = False,
        kernel_size: int = 3,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.third_up = third_up
        self.scale_factor = scale_factor
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, kernel_size, padding=padding
            )

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels

        if self.dims == 3:
            t_factor = 1 if not self.third_up else self.scale_factor
            x = F.interpolate(
                x,
                (
                    t_factor * x.shape[2],
                    x.shape[3] * self.scale_factor,
                    x.shape[4] * self.scale_factor,
                ),
                mode="nearest",
            )
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

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
    def __init__(
            self, 
            attention_resolutions: int = [ 4, 2, 1],
            model_channels=320, 
            out_channels=4, 
            in_channels=4,
            channel_mult: Union[List, Tuple] = [1, 2, 4, 4], 
            dims=2, 
            num_classes: Optional[Union[int, str]] = 'sequential',
            addition_time_embed_dim=256,
            projection_class_embeddings_input_dim=1280, 
            context_dim=768,  
            num_res_blocks: int = 2,
            dropout: float = 0.0,
            conv_resample: bool = True,
            use_checkpoint: bool = False,
            n_heads: int = -1,
            n_heads_channels: int = -1,
            n_heads_upsample: int = -1,
            use_scale_shift_norm: bool = False,
            resblock_updown: bool = False,
            transformer_depth: int = 1,
            disable_self_attentions: Optional[List[bool]] = None,
            num_attention_blocks: Optional[List[int]] = None,
            disable_middle_transformer: bool = False,
            use_linear_in_transformer: bool = False,
            spatial_transformer_attn_type: str = "softmax",
            ):
        super().__init__()
        self.dims = dims
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_classes = num_classes
        self.context_dim = context_dim
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.n_heads = n_heads
        self.n_heads_channels = n_heads_channels
        self.n_heads_upsample = n_heads_upsample

        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = transformer_depth[-1]

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )


  
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
           self.add_time_proj = Timesteps(addition_time_embed_dim, True, 0)
            
           self.add_embedding = TimestepEmbedding(
                projection_class_embeddings_input_dim + 6 * addition_time_embed_dim,
                time_embed_dim,
                time_embed_dim  # Intermediate dimension
            )
        
        self.input_blocks = nn.ModuleList(
            [
                SwitchSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    UNET_ResidualBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if n_heads_channels == -1:
                        d_head = ch // n_heads
                    else:
                        n_heads = ch // n_heads_channels
                        d_head = n_heads_channels

                    if context_dim is not None and exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                n_heads,
                                d_head,
                                depth=transformer_depth[level],
                                disable_self_attn=disabled_sa,
                                context_dim=context_dim,
                                attn_type=spatial_transformer_attn_type,
                                use_linear=use_linear_in_transformer,
                            )
                        )
                self.input_blocks.append(SwitchSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    SwitchSequential(
                        UNET_ResidualBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if n_heads_channels == -1:
            d_head = ch // n_heads
        else:
            n_heads = ch // n_heads_channels
            d_head = n_heads_channels

        self.middle_block = SwitchSequential(
            UNET_ResidualBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(
                ch,
                n_heads,
                d_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,        
                attn_type=spatial_transformer_attn_type,
                use_linear=use_linear_in_transformer,
            )
            if not disable_middle_transformer
            else th.nn.Identity(),
            UNET_ResidualBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    UNET_ResidualBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if n_heads_channels == -1:
                        d_head = ch // n_heads
                    else:
                        n_heads = ch // n_heads_channels
                        d_head = n_heads_channels

                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False


                    if (
                        not exists(num_attention_blocks)
                        or i < num_attention_blocks[level]
                    ):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                n_heads,
                                d_head,
                                depth=transformer_depth[level],
                                disable_self_attn=disabled_sa,
                                context_dim=context_dim,                          
                                attn_type=spatial_transformer_attn_type,
                                use_linear=use_linear_in_transformer,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        UNET_ResidualBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(SwitchSequential(*layers))
                self._feature_size += ch
        
        # self.final = UNET_OutputLayer(320, 4)
        self.out = nn.Sequential(
            normalization(self.model_channels),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, self.model_channels, self.out_channels, 3, padding=1)),
        )

    def forward(
            self, 
            latent: th.Tensor, 
            context: Optional[th.Tensor] = None, 
            time: Optional[th.Tensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None, 
            ) -> th.Tensor:
        
         # Ensure consistent dtype
        model_dtype = next(self.parameters()).dtype
        latent = latent.to(dtype=model_dtype)
        if time is not None:
            time = time.to(dtype=model_dtype)
            # time_emb = self.time_embedding(time)
            t_emb = timestep_embedding(time, self.model_channels, repeat_only=False)
            t_emb = t_emb.to(dtype=model_dtype)
            emb = self.time_embed(t_emb)

        # Process SDXL micro-conditioning
        if added_cond_kwargs is not None:
            text_embeds = added_cond_kwargs["text_embeds"].to(dtype=model_dtype)
            time_ids = added_cond_kwargs["time_ids"].to(dtype=model_dtype)
            
            # Embed time_ids (6 values -> 6*256=1536)
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.to(dtype=model_dtype)
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            
            # Combine with pooled text embeddings
            aug_embeds = torch.cat([text_embeds, time_embeds], dim=-1)
            add_embed = self.add_embedding(aug_embeds)
            emb = emb + add_embed.to(dtype=emb.dtype)

        x = latent
        skip_connections = []
        
        # Process input_blocks
        for encoder in self.input_blocks:
            x = encoder(x, context, emb)
            skip_connections.append(x)

        # Process middle_blocks
        x = self.middle_block(x, context, emb)
        
        # Process output_blocks
        for decoder in self.output_blocks:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = decoder(x, context, emb)
        
        return self.out(x)

class Timesteps(nn.Module):
    def __init__(self, num_channels, flip_sin_to_cos, downscale_freq_shift):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        input_dtype = timesteps.dtype
        t = timesteps.float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * (torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device) - half_dim
        ) / (half_dim + self.downscale_freq_shift)
        emb = t[:, None] * exponent.exp()
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        
        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        
        if self.num_channels % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb.to(dtype=input_dtype)

class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, projection_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, projection_dim)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(projection_dim, out_features)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.activation(sample)
        sample = self.linear_2(sample)
        return sample
