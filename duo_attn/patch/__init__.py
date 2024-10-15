from .llama import (
    enable_llama_duo_attention_training,
    enable_llama_duo_attention_eval,
    get_llama_full_attention_heads,
    set_llama_full_attention_heads,
    map_llama_full_attention_heads,
)

from .mistral import (
    enable_mistral_duo_attention_training,
    enable_mistral_duo_attention_eval,
    get_mistral_full_attention_heads,
    set_mistral_full_attention_heads,
    map_mistral_full_attention_heads,
)

import numpy as np
import os
import torch


def enable_duo_attention_training(
    model,
    sink_size,
    recent_size,
    max_length,
    initial_value=1.0,
    enable_ulysses_attention=False,
    streaming_attn_implementation="blocksparse",
):
    print(
        f"Enabling DuoAttention training using {streaming_attn_implementation} imlementation"
    )
    if "llama" in model.config.model_type:
        enable_llama_duo_attention_training(
            model,
            sink_size,
            recent_size,
            max_length,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            streaming_attn_implementation=streaming_attn_implementation,
        )
    elif "mistral" in model.config.model_type or "mixtral" in model.config.model_type:
        enable_mistral_duo_attention_training(
            model,
            sink_size,
            recent_size,
            max_length,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            streaming_attn_implementation=streaming_attn_implementation,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def enable_duo_attention_eval(
    model,
    full_attention_heads,
    sink_size,
    recent_size,
):
    print(
        f"Enabling DuoAttention evaluation using sink size {sink_size} and recent size {recent_size}"
    )
    if "llama" in model.config.model_type:
        enable_llama_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    elif "mistral" in model.config.model_type or "mixtral" in model.config.model_type:
        enable_mistral_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def get_full_attention_heads(model):
    if "llama" in model.config.model_type:
        return get_llama_full_attention_heads(model)
    elif "mistral" in model.config.model_type or "mixtral" in model.config.model_type:
        return get_mistral_full_attention_heads(model)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def set_full_attention_heads(model, full_attention_heads):
    if "llama" in model.config.model_type:
        model = set_llama_full_attention_heads(model, full_attention_heads)
    elif "mistral" in model.config.model_type or "mixtral" in model.config.model_type:
        model = set_mistral_full_attention_heads(model, full_attention_heads)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")
    return model


def map_full_attention_heads(model, func):
    if "llama" in model.config.model_type:
        return map_llama_full_attention_heads(model, func)
    elif "mistral" in model.config.model_type or "mixtral" in model.config.model_type:
        return map_mistral_full_attention_heads(model, func)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def load_full_attention_heads(load_dir, filename="full_attention_heads.tsv"):
    full_attention_heads = np.loadtxt(
        os.path.join(load_dir, filename),
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    full_attention_heads = torch.tensor(full_attention_heads, dtype=torch.float32)
    return full_attention_heads
