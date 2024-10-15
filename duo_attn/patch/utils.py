import torch
import numpy as np
import os


@torch.no_grad()
def reorder_linear_weights(
    linear_module: torch.nn.Linear,
    full_attention_heads: torch.Tensor,
    repeat_num,
    reorder_channel,
):
    assert reorder_channel in ["in", "out"]
    full_attention_heads = torch.repeat_interleave(
        full_attention_heads, repeats=repeat_num
    ).to(linear_module.weight.device)
    full_attn_mask = full_attention_heads > 0.5
    if reorder_channel == "in":
        weight1 = linear_module.weight.data[:, full_attn_mask]
        weight2 = linear_module.weight.data[:, ~full_attn_mask]
        reordered_weight = torch.cat([weight1, weight2], dim=1)
    else:
        weight1 = linear_module.weight.data[full_attn_mask, :]
        weight2 = linear_module.weight.data[~full_attn_mask, :]
        reordered_weight = torch.cat([weight1, weight2], dim=0)
    linear_module.weight.data = reordered_weight
    return linear_module


@torch.no_grad()
def reorder_full_attn_heads(
    full_attention_heads: torch.Tensor,
):
    full_attn_mask = full_attention_heads > 0.5
    num_full_attn_heads = full_attn_mask.sum().item()
    full_attention_heads[:num_full_attn_heads] = 1
    full_attention_heads[num_full_attn_heads:] = 0
    return full_attention_heads
