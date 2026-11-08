"""TinyGPT-style causal language model (~15M parameters with 32K vocabulary)."""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class TinyGPTConfig:
    """Configuration class for TinyGPTForCausalLM."""

    vocab_size: int = 32000
    hidden_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    max_position_embeddings: int = 2048
    layer_norm_epsilon: float = 1e-5
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    tie_word_embeddings: bool = True
    bias: bool = False


class TinyGPTAttention(nn.Module):
    """Multi-head causal self-attention with separate Q, K, V, O projections."""

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})."
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attn_pdrop = config.attn_pdrop

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()

        q = (
            self.q_proj(x)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=True,
        )
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.hidden_size)
        )
        return self.resid_dropout(self.o_proj(attn_out))


class TinyGPTMLP(nn.Module):
    """Feed-forward network with GELU activation for TinyGPT."""

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.bias
        )
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.bias
        )
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class TinyGPTBlock(nn.Module):
    """Pre-LayerNorm Transformer block for TinyGPT."""

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = TinyGPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = TinyGPTMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPTForCausalLM(nn.Module):
    """
    TinyGPT causal language model (~15.02M parameters with 32K vocab, d=256, L=8, H=8).
    Compatible with HuggingFace-style `model(input_ids=..., labels=...)` calls.
    """

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [TinyGPTBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.init_weights()

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def init_weights(self, std: float = 0.02) -> None:
        """Initialize weights following standard GPT-2 initialization with residual depth scaling."""
        with torch.no_grad():
            nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)
            nn.init.normal_(self.wpe.weight, mean=0.0, std=std)

            resid_std = std / math.sqrt(2.0 * max(1, self.config.num_hidden_layers))

            for block in self.layers:
                nn.init.normal_(block.attn.q_proj.weight, mean=0.0, std=std)
                nn.init.normal_(block.attn.k_proj.weight, mean=0.0, std=std)
                nn.init.normal_(block.attn.v_proj.weight, mean=0.0, std=std)
                nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=resid_std)

                nn.init.normal_(block.mlp.c_fc.weight, mean=0.0, std=std)
                nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=resid_std)

                for proj in (
                    block.attn.q_proj,
                    block.attn.k_proj,
                    block.attn.v_proj,
                    block.attn.o_proj,
                    block.mlp.c_fc,
                    block.mlp.c_proj,
                ):
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

                nn.init.ones_(block.ln_1.weight)
                nn.init.zeros_(block.ln_1.bias)
                nn.init.ones_(block.ln_2.weight)
                nn.init.zeros_(block.ln_2.bias)

            nn.init.ones_(self.ln_f.weight)
            nn.init.zeros_(self.ln_f.bias)

            if not self.config.tie_word_embeddings:
                nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        bsz, seq_len = input_ids.size()
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_position_embeddings "
                f"({self.config.max_position_embeddings})."
            )

        positions = torch.arange(
            0, seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        x = self.drop(self.embed_tokens(input_ids) + self.wpe(positions))

        for block in self.layers:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

