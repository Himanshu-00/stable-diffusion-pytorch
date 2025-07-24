# clip.py contains the implementation of the CLIPEmbedding, CLIPLayer and CLIP classes.
import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput
from attention import CLIPAttention, ACTIVATIONS_FN
import json
import re
import torch
from transformers import CLIPTokenizer


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0




@dataclass
class CLIPTextModelOutput(ModelOutput):
    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config["hidden_size"]

        self.token_embedding = nn.Embedding(config["vocab_size"], embed_dim)
        self.position_embedding = nn.Embedding(config["max_position_embeddings"], embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config["max_position_embeddings"]).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings

class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACTIVATIONS_FN[config["hidden_act"]]
        self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["hidden_size"]
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config["layer_norm_eps"])
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config["layer_norm_eps"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ) -> Tuple[torch.FloatTensor]:
       
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)


        return outputs

class CLIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config["num_hidden_layers"])])

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
       
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config["output_hidden_states"]
        )
       

        encoder_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )

            hidden_states = layer_outputs[0]


        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )

class CLIPTextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config["hidden_size"]
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config["layer_norm_eps"])
        self.eos_token_id = config["eos_token_id"]
      

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, CLIPTextModelOutput]:
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config["output_hidden_states"]
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)


        return CLIPTextModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class CLIPTextModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_model = CLIPTextTransformer(self.config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, CLIPTextModelOutput]:


        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )

class SDTokenizer:
    def __init__(self, tokenizer_path, max_len=77, debug=False):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        self.debug = debug
        
        # Cache compiled regex for better performance
        self._weight_pattern = re.compile(r'\(([^:)]+):([0-9.]+)\)')
        
        # Get special tokens automatically
        empty = self.tokenizer('')["input_ids"]
        self.start_token = empty[0]
        self.end_token = empty[1] if len(empty) > 1 else empty[0]
        
        # Create inverse vocabulary for debugging
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def _has_weights(self, text):
        """Fast weight detection"""
        if not text:
            return False
        return self._weight_pattern.search(text) is not None
    
    def get_raw_tokenizer(self):
        """Access to raw CLIP tokenizer"""
        return self.tokenizer


    def tokenize_with_weights(self, text):
        """Optimized weight parsing with fast path for non-weighted text"""
        
        # Quick weight detection
        if not self._has_weights(text):
            return self._process_simple(text)
        
        return self._process_weighted(text)
    
    def _has_weights(self, text):
        """Fast weight detection"""
        return self._weight_pattern.search(text) is not None
    
    def _process_simple(self, text):
        """Fast path for non-weighted text"""
        tokens = [(self.start_token, 1.0)]
        
        if text.strip():
            token_ids = self.tokenizer(text.strip())["input_ids"][1:-1]
            tokens.extend([(t, 1.0) for t in token_ids])
        
        tokens.append((self.end_token, 1.0))
        # while len(tokens) < self.max_len:
        #     tokens.append((self.end_token, 1.0))
        tokens_len = len(tokens)
        if tokens_len < self.max_len:
            tokens.extend([(self.end_token, 1.0)] * (self.max_len - tokens_len))
        
        return tokens[:self.max_len]
    
    def _process_weighted(self, text):
        """Complex parsing for weighted text (your existing logic)"""
        weighted_terms = []
        
        def extract_weighted_term(match):
            term = match.group(1).strip()
            try:
                weight = float(match.group(2))
                if weight <= 0:
                    if self.debug:
                        print(f"Warning: Invalid weight {weight} for '{term}', using 1.0")
                    weight = 1.0
            except ValueError:
                if self.debug:
                    print(f"Warning: Invalid weight format for '{term}', using 1.0")
                weight = 1.0
            
            weighted_terms.append((term, weight))
            return f"__WEIGHTED_{len(weighted_terms)-1}__"
        
        # Replace weighted terms with placeholders
        processed_text = self._weight_pattern.sub(extract_weighted_term, text)
        
        if self.debug:
            print(f"=== COMMA-AWARE PARSING ===")
            print(f"Original: {text}")
            print(f"Weighted terms found: {weighted_terms}")
            print(f"After extraction: {processed_text}")
        
        # Your existing comma processing logic
        parts = processed_text.split(',')
        tokens = [(self.start_token, 1.0)]
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            weighted_match = re.search(r'__WEIGHTED_(\d+)__', part)
            
            if weighted_match:
                idx = int(weighted_match.group(1))
                term, weight = weighted_terms[idx]
                
                term_tokens = self.tokenizer(term.strip())["input_ids"][1:-1]
                tokens.extend([(t, weight) for t in term_tokens])
                
                if self.debug:
                    print(f"Weighted term '{term}': {len(term_tokens)} tokens with weight {weight}")
                
                remaining = part.replace(weighted_match.group(0), '').strip()
                if remaining:
                    remaining_tokens = self.tokenizer(remaining)["input_ids"][1:-1]
                    tokens.extend([(t, 1.0) for t in remaining_tokens])
            else:
                if part:
                    regular_tokens = self.tokenizer(part)["input_ids"][1:-1]
                    tokens.extend([(t, 1.0) for t in regular_tokens])
                    if self.debug:
                        print(f"Regular term '{part}': {len(regular_tokens)} tokens with weight 1.0")
        
        tokens.append((self.end_token, 1.0))
        while len(tokens) < self.max_len:
            tokens.append((self.end_token, 1.0))
        
        final_tokens = tokens[:self.max_len]
        
        if self.debug:
            weights_sample = [w for _, w in final_tokens[:10]]
            print(f"Final weights sample: {weights_sample}")
            print(f"=== END COMMA-AWARE PARSING ===\n")
        
        return final_tokens
    
    def untokenize(self, token_weight_pair):
        """Convert tokens back to text for debugging"""
        return [(token_id, self.inv_vocab.get(token_id, '<unk>')) for token_id, weight in token_weight_pair]
    
    def state_dict(self):
        """Return state dictionary"""
        return {}
    
    def encode(self, text):
        """Simple encode without weights"""
        return self.tokenizer(text, padding="max_length", max_length=self.max_len, 
                            truncation=True, return_tensors="pt")

class SD1Tokenizer:
    def __init__(self, tokenizer_class=SDTokenizer):
        # Direct, simple approach
        self.clip_l = tokenizer_class(
            tokenizer_path="sd1_tokenizer_",
        )

    def has_weights(self, text):
        """Check if text contains weights"""
        return self.clip_l._has_weights(text)
    
    def get_raw_tokenizer(self):
        """Access to raw CLIP tokenizer"""
        return self.clip_l.tokenizer
    
    def tokenize_with_weights(self, text):
        # Direct method call - no dynamic attributes needed
        return {"l": self.clip_l.tokenize_with_weights(text)}
    
    def untokenize(self, token_weight_pair):
        return self.clip_l.untokenize(token_weight_pair)
    
    def state_dict(self):
        return self.clip_l.state_dict()

class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    
    def __init__(
        self,
        freeze=True,
        layer="last",
        layer_idx=None,
        dtype=torch.float16,
    ):
        super().__init__()

        with open("stable_diffusion/sd1_clip_config.json") as f:
            self.config = json.load(f)
        self.transformer = CLIPTextModel(self.config)

        #Backward-compatible with old checkpoints 
        emb = self.transformer.text_model.embeddings
        if hasattr(emb, 'position_ids'):
            emb._non_persistent_buffers_set.discard('position_ids')
          
            
        if freeze:
            self.freeze()
        self.dtype = dtype
        self.layer = layer
        self.layer_idx = layer_idx 
        self.num_layers = self.config["num_hidden_layers"]
        if layer == "hidden":
            if layer_idx is None:
                raise ValueError("layer_idx must be specified for 'hidden' mode")
            if not (0 <= layer_idx < self.num_layers):
                raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.num_layers-1}]")
        self.special_tokens = {
            "start": 49406,  # CLIP-L values
            "end": 49407,
            "pad": 49407
        }
    
    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, tokens, clip_skip=0, output_hidden_states=True):

        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=output_hidden_states
        )
        z = outputs
         # Handle clip_skip functionality
        if clip_skip > 0:
            # Validate skip depth
            if clip_skip >= self.num_layers:
                raise ValueError(f"clip_skip ({clip_skip}) exceeds total layers ({self.num_layers})")
                      
            # Get all hidden states (output_hidden_states=True ensures this exists)
            hidden_states = outputs.hidden_states  

            # Reverse index: -1 is final layer, -2 is penultimate, etc.
            selected_layer = hidden_states[-clip_skip-1]  

            z = selected_layer

            z = self.transformer.text_model.final_layer_norm(z).to(dtype=self.dtype)
        else:
            if self.layer == "last":
                z = outputs.last_hidden_state

            elif self.layer == "hidden":
                # Get ALL hidden states including embeddings
                hidden_states = outputs.hidden_states
                
                # Calculate adjusted index (embeddings are index 0)
                adjusted_idx = self.layer_idx + 1
                
                # Validate adjusted index
                if adjusted_idx >= len(hidden_states):
                    raise ValueError(f"Layer index {self.layer_idx} too high")
                
                # Get specific layer output
                z = hidden_states[adjusted_idx]

            else:
                z = outputs.last_hidden_state

        return z
    
    def encode(self, text, clip_skip=0):
        return self(text, clip_skip=clip_skip)

class SDXLTokenizer:
    def __init__(self):
        
        self.clip_g = SDTokenizer(
            tokenizer_path="sdxl_tokenizer_", 
        )

    def has_weights(self, text):
        """Check if text contains weights"""
        return self.clip_g._has_weights(text)
    
    def get_raw_tokenizer(self):
        """Access to raw CLIP tokenizer"""
        return self.clip_g.tokenizer
    
    def tokenize_with_weights(self, text):
        # Simple, direct calls
        return {
            "g": self.clip_g.tokenize_with_weights(text)
        }
    
    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)
    
    def state_dict(self):
        return {}

class CLIPGEmbedder(nn.Module):
    """CLIP-G embedder for SDXL compatibility with custom transformer"""
    
    def __init__(self, freeze=True, dtype=torch.float16, config_path="stable_diffusion/clip_config_bigg.json"):
        super().__init__()
        # Load config from JSON
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Initialize transformer with config dict
        self.transformer = CLIPTextModel(self.config)
        
        # Backward compatibility fix
        emb = self.transformer.text_model.embeddings
        if hasattr(emb, 'position_ids'):
            emb._non_persistent_buffers_set.discard('position_ids')
        
        # Use config values
        self.pad_token_id = self.config["pad_token_id"]
        self.eos_token_id = self.config["eos_token_id"]
        
        # SDXL projection
        self.text_projection = nn.Linear(
            self.config["hidden_size"], 
            self.config["projection_dim"],  # 1280 for SDXL
            bias=False
        )

        self.dtype = dtype
        
        if freeze:
            self.freeze()
    
    def freeze(self):
        self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, tokens, output_hidden_states=None):        
        # Full transformer call with position IDs
        output = self.transformer(
            input_ids=tokens,
            output_hidden_states=output_hidden_states
        )
        
        hidden_states = output.hidden_states
        last_hidden_state = output.last_hidden_state
        
        
        penultimate_hidden_state = hidden_states[-2] if len(hidden_states) > 1 else last_hidden_state
        eos_positions = (tokens == self.eos_token_id).int().argmax(dim=-1)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0]), 
            eos_positions
        ]
        # Project pooled output
        text_embeds = self.text_projection(pooled_output)

        return penultimate_hidden_state.to(self.dtype), text_embeds.to(self.dtype)
    
    