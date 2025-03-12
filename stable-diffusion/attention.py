# attention.py contains the implementation of the SelfAttention and CrossAttention classes.

import torch
from torch import nn
from torch.nn import functional as F
import math
from einops import rearrange, repeat
from inspect import isfunction
from torch import nn, einsum



def exists(val):
    return val is not None 

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        # Store scale as an attribute for efficiency
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x, causal_mask=False):
        # x: # (Batch_Size, Seq_Len, Dim)

        
        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = x.shape

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).permute(0, 2, 1, 3)
        k = k.view(interim_shape).permute(0, 2, 1, 3)
        v = v.view(interim_shape).permute(0, 2, 1, 3)

        # Apply scaling to q
        q = q * self.scale

        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = torch.matmul(q, k.transpose(-1, -2))
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.finfo(weight.dtype).max) 
        
        # # Divide by d_k (Dim / H). 
        # # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)


        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1) 

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.permute(0, 2, 1, 3).contiguous()

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.view(batch_size, sequence_length, d_embed)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output) 
        
        # (Batch_Size, Seq_Len, Dim)
        return output

# class CrossAttention(nn.Module):
#     def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
#         super().__init__()
#         self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
#         self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
#         self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
#         self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
#         self.n_heads = n_heads
#         self.d_head = d_embed // n_heads
#         # Store scale as an attribute for efficiency
#         self.scale = 1.0 / math.sqrt(self.d_head)
    
#     def forward(self, x, y):
#         # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
#         # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

#         batch_size, sequence_length, _ = x.shape
#         # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
#         interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
#         # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
#         q = self.q_proj(x)
#         # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
#         k = self.k_proj(y)
#         # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
#         v = self.v_proj(y)

#         # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
#         q = q.view(batch_size,sequence_length, self.n_heads, self.d_head).permute(0, 2, 1, 3)
#         # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
#         k = k.view(interim_shape).permute(0, 2, 1, 3)
#         # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
#         v = v.view(interim_shape).permute(0, 2, 1, 3)

#         # Apply scaling to q
#         q = q * self.scale
        
#         # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
#         weight = torch.matmul(q , k.transpose(-1, -2))
        
#         # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
#         weight = F.softmax(weight, dim=-1)
        
#         # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
#         output = torch.matmul(weight, v)
        
#         # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
#         output = output.permute(0, 2, 1, 3).contiguous()
        
#         # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
#         output = output.view(batch_size, sequence_length, -1)
        
#         # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
#         output = self.out_proj(output)

#         # (Batch_Size, Seq_Len_Q, Dim_Q)
#         return output
    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)    
