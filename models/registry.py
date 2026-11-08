"""Model architecture registry, configuration loading, and weight initialization."""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from .tinygpt import TinyGPTConfig, TinyGPTForCausalLM

MODEL_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
REPO_ROOT = Path(__file__).resolve().parent.parent

PRESET_HPARAMS_FILES: Dict[str, Path] = {
    "15m": MODEL_CONFIGS_DIR / "15M_model_hparams.json",
    "15M": MODEL_CONFIGS_DIR / "15M_model_hparams.json",
    "178m": MODEL_CONFIGS_DIR / "178M_model_hparams.json",
    "178M": MODEL_CONFIGS_DIR / "178M_model_hparams.json",
    "512m": MODEL_CONFIGS_DIR / "512M_model_hparams.json",
    "512M": MODEL_CONFIGS_DIR / "512M_model_hparams.json",
}

MODEL_REGISTRY: Dict[str, Callable[[Dict[str, Any], int], nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model builder function by architecture name."""

    def decorator(fn: Callable[[Dict[str, Any], int], nn.Module]):
        MODEL_REGISTRY[name.lower()] = fn
        return fn

    return decorator


def resolve_hparams_path(
    hparams_file: Optional[str] = None,
    model_size: Optional[str] = None,
) -> Path:
    """Resolve model hyperparameters path from an explicit file path or size preset."""
    if hparams_file:
        candidate = Path(os.path.expanduser(hparams_file))
        if candidate.is_file():
            return candidate
        repo_candidate = REPO_ROOT / hparams_file
        if repo_candidate.is_file():
            return repo_candidate
        if hparams_file in PRESET_HPARAMS_FILES:
            return PRESET_HPARAMS_FILES[hparams_file]
        raise FileNotFoundError(f"Model hparams file not found: {hparams_file}")

    if model_size:
        if model_size in PRESET_HPARAMS_FILES:
            return PRESET_HPARAMS_FILES[model_size]
        raise ValueError(
            f"Unknown model_size '{model_size}'. Available presets: ['15M', '178M', '512M']"
        )

    return PRESET_HPARAMS_FILES["178M"]


def load_model_hparams(
    hparams_file: Optional[str] = None,
    model_size: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load model hyperparameters from JSON and apply optional overrides."""
    path = resolve_hparams_path(hparams_file=hparams_file, model_size=model_size)
    with open(path, "r", encoding="utf-8") as f:
        hparams = json.load(f)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                hparams[k] = v
    return hparams


def build_tokenizer(
    tokenizer_name: str,
    use_fast: bool = False,
    revision: Optional[str] = None,
):
    """Load and configure the HuggingFace tokenizer."""
    kwargs: Dict[str, Any] = {"use_fast": use_fast}
    if revision is not None:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def init_torchtitan_weights(model: LlamaForCausalLM, std: float = 0.02) -> None:
    """
    Apply Torchtitan-style initialization to LlamaForCausalLM.
    Preserves the exact initialization logic from the official IsoLoCo train.py.
    """
    with torch.no_grad():
        nn.init.normal_(model.model.embed_tokens.weight, mean=0.0, std=std)

        for layer_idx, layer in enumerate(model.model.layers):
            depth_std = std / (2 * (layer_idx + 1)) ** 0.5

            nn.init.normal_(layer.self_attn.q_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.self_attn.k_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.self_attn.v_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=depth_std)

            nn.init.normal_(layer.mlp.gate_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.mlp.up_proj.weight, mean=0.0, std=depth_std)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=depth_std)

            nn.init.ones_(layer.input_layernorm.weight)
            nn.init.ones_(layer.post_attention_layernorm.weight)

        nn.init.ones_(model.model.norm.weight)

        if not model.config.tie_word_embeddings:
            final_out_std = model.config.hidden_size**-0.5
            nn.init.trunc_normal_(
                model.lm_head.weight,
                mean=0.0,
                std=final_out_std,
                a=-3 * final_out_std,
                b=3 * final_out_std,
            )


@register_model("llama")
def build_llama_model(
    model_hparams: Dict[str, Any], vocab_size: int
) -> LlamaForCausalLM:
    """Build a LlamaForCausalLM instance from model hyperparameters."""
    model_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=model_hparams["hidden_size"],
        num_hidden_layers=model_hparams["num_hidden_layers"],
        num_attention_heads=model_hparams["num_attention_heads"],
        intermediate_size=model_hparams["intermediate_size"],
        num_key_value_heads=model_hparams["num_key_value_heads"],
        activation_function=model_hparams["activation_function"],
        max_position_embeddings=model_hparams["max_position_embeddings"],
    )
    return LlamaForCausalLM(model_config)


@register_model("tinygpt")
def build_tinygpt_model(
    model_hparams: Dict[str, Any], vocab_size: int
) -> TinyGPTForCausalLM:
    """Build a ~15M TinyGPTForCausalLM instance from model hyperparameters."""
    model_config = TinyGPTConfig(
        vocab_size=vocab_size,
        hidden_size=model_hparams.get("hidden_size", 256),
        num_hidden_layers=model_hparams.get("num_hidden_layers", 8),
        num_attention_heads=model_hparams.get("num_attention_heads", 8),
        intermediate_size=model_hparams.get("intermediate_size", 1024),
        max_position_embeddings=model_hparams.get("max_position_embeddings", 2048),
        layer_norm_epsilon=model_hparams.get("layer_norm_epsilon", 1e-5),
        resid_pdrop=model_hparams.get("resid_pdrop", 0.0),
        embd_pdrop=model_hparams.get("embd_pdrop", 0.0),
        attn_pdrop=model_hparams.get("attn_pdrop", 0.0),
        tie_word_embeddings=model_hparams.get("tie_word_embeddings", True),
        bias=model_hparams.get("bias", False),
    )
    return TinyGPTForCausalLM(model_config)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap compiled or DDP-wrapped models to access the underlying module."""
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


def attach_param_names(model: nn.Module) -> None:
    """Attach parameter names to Parameter objects for method-level filtering."""
    for name, param in model.named_parameters():
        param.param_name = name


def build_model(
    model_hparams: Dict[str, Any],
    vocab_size: int,
    architecture: str = "llama",
    torchtitan_init: bool = False,
    device: Optional[torch.device] = None,
    use_compile: bool = False,
) -> nn.Module:
    """Construct, initialize, move to device, optionally compile, and tag parameter names."""
    arch_key = architecture.lower()
    if arch_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture '{architecture}'. Available: {sorted(MODEL_REGISTRY.keys())}"
        )

    model = MODEL_REGISTRY[arch_key](model_hparams, vocab_size)

    if torchtitan_init and isinstance(model, LlamaForCausalLM):
        init_torchtitan_weights(model)

    if device is not None:
        model.to(device)

    if use_compile:
        model = torch.compile(model)

    attach_param_names(model)
    return model

