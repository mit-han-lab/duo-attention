from typing import Optional, Tuple
import os
import torch
from torch import nn

from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM,
    MistralModel,
    repeat_kv,
    apply_rotary_pos_emb,
    CausalLMOutputWithPast,
    List,
    Union,
    CrossEntropyLoss,
    BaseModelOutputWithPast,
)
import types
from .utils import (
    reorder_linear_weights,
    reorder_full_attn_heads,
)
from .streaming_attn import (
    generate_streaming_mask,
    streaming_attn_sdpa,
    generate_streaming_info_blocksparse_flash_attn,
    streaming_attn_blocksparse_flash_attn,
)

from .static_kv_cache import (
    DuoAttentionStaticKVCache,
    enable_duo_attention_static_kv_cache_for_mistral,
)
from .tuple_kv_cache import enable_tuple_kv_cache_for_mistral
from .flashinfer_utils import apply_rope_inplace, enable_flashinfer_rmsnorm

from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from duo_attn.ulysses import UlyssesAttention


def mistral_duo_attention_forward_two_way(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz_x_2, q_len, _ = hidden_states.size()

    assert bsz_x_2 % 2 == 0

    bsz = bsz_x_2 // 2

    full_hidden_states = hidden_states[:bsz]
    streaming_hidden_states = hidden_states[bsz:]

    with torch.no_grad():
        full_query_states = self.q_proj(full_hidden_states)
        full_key_states = self.k_proj(full_hidden_states)
        full_value_states = self.v_proj(full_hidden_states)
        full_query_states = full_query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        full_key_states = full_key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        full_value_states = full_value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

    streaming_query_states = self.q_proj(streaming_hidden_states)
    streaming_key_states = self.k_proj(streaming_hidden_states)
    streaming_value_states = self.v_proj(streaming_hidden_states)
    streaming_query_states = streaming_query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    )
    streaming_key_states = streaming_key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )
    streaming_value_states = streaming_value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )

    cos, sin = self.rotary_emb(full_value_states, position_ids)

    with torch.no_grad():
        full_query_states, full_key_states = apply_rotary_pos_emb(
            full_query_states,
            full_key_states,
            cos,
            sin,
            unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
        )
        full_attn_output = self.full_attn_func(
            full_query_states,
            full_key_states,
            full_value_states,
            causal=True,
            dropout_p=0.0,
        )

    streaming_query_states, streaming_key_states = apply_rotary_pos_emb(
        streaming_query_states,
        streaming_key_states,
        cos,
        sin,
        unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
    )

    streaming_attn_output = self.streaming_attn_func(
        streaming_query_states,
        streaming_key_states,
        streaming_value_states,
        self.streaming_mask,
    )

    full_attention_heads = (
        self.full_attention_heads.clamp(0, 1)
        .view(1, 1, self.num_key_value_heads, 1, 1)
        .expand(1, 1, self.num_key_value_heads, self.num_key_value_groups, 1)
        .reshape(1, 1, self.num_heads, 1)
    )

    streaming_attn_output = (
        1 - full_attention_heads
    ) * streaming_attn_output + full_attention_heads * full_attn_output

    with torch.no_grad():
        full_attn_output = full_attn_output.reshape(bsz, q_len, self.hidden_size)
        full_attn_output = self.o_proj(full_attn_output)

    streaming_attn_output = streaming_attn_output.reshape(bsz, q_len, self.hidden_size)
    streaming_attn_output = self.o_proj(streaming_attn_output)

    attn_output = torch.cat([full_attn_output, streaming_attn_output], dim=0)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def mistral_duo_attention_forward_one_way_reordered(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )

    # new data structure for past_key_value
    # past_key_value = (full_KV, streaming_KV)
    # full_KV: (2 x bsz, num_full_key_value_heads, full_kv_seq_len, head_dim)
    # streaming_KV: (2 x bsz, num_streaming_key_value_heads, cache_size, head_dim)

    kv_seq_len = key_states.shape[1]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[2]

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
    )

    if not hasattr(self, "full_attn_head_mask") or self.full_attn_head_mask is None:
        self.full_attn_head_mask = self.full_attention_heads > 0.5
        self.num_full_attn_head = self.full_attn_head_mask.sum().item()
        self.num_streaming_attn_head = (
            self.num_key_value_heads - self.num_full_attn_head
        )

        self.num_full_query_head = self.num_full_attn_head * self.num_key_value_groups
        self.num_streaming_query_head = self.num_heads - self.num_full_query_head

    full_key_states = key_states[:, :, : self.num_full_attn_head, :]
    full_value_states = value_states[:, :, : self.num_full_attn_head, :]

    streaming_key_states = key_states[:, :, self.num_full_attn_head :, :]
    streaming_value_states = value_states[:, :, self.num_full_attn_head :, :]

    if past_key_value is not None:
        # reuse k, v, self_attention
        past_full_KV = past_key_value[0].transpose(1, 2)
        past_streaming_KV = past_key_value[1].transpose(1, 2)

        past_full_key_states = past_full_KV[:bsz]
        past_full_value_states = past_full_KV[bsz:]

        full_key_states = torch.cat([past_full_key_states, full_key_states], dim=1)
        full_value_states = torch.cat(
            [past_full_value_states, full_value_states], dim=1
        )

        past_streaming_key_states = past_streaming_KV[:bsz]
        past_streaming_value_states = past_streaming_KV[bsz:]

        streaming_key_states = torch.cat(
            [past_streaming_key_states, streaming_key_states], dim=1
        )
        streaming_value_states = torch.cat(
            [past_streaming_value_states, streaming_value_states], dim=1
        )

    if q_len == kv_seq_len:
        # pre-filling: use flash attention
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=0.0,
        )
    else:
        # decoding or continous filling
        if self.num_full_attn_head > 0:
            full_query_states = query_states[:, :, : self.num_full_query_head, :]

            full_attn_output = flash_attn_func(
                full_query_states,
                full_key_states,
                full_value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            full_attn_output = None

        if self.num_streaming_attn_head > 0:
            streaming_query_states = query_states[:, :, self.num_full_query_head :, :]

            streaming_attn_output = flash_attn_func(
                streaming_query_states,
                streaming_key_states,
                streaming_value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            streaming_attn_output = None

        if full_attn_output is None:
            attn_output = streaming_attn_output
        elif streaming_attn_output is None:
            attn_output = full_attn_output
        else:
            attn_output = torch.cat([full_attn_output, streaming_attn_output], dim=2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if streaming_key_states.shape[1] > self.recent_size + self.sink_size:
        recent_key_states = streaming_key_states[:, -self.recent_size :, :, :].clone()
        streaming_key_states[
            :, self.sink_size : self.sink_size + self.recent_size, :, :
        ].copy_(recent_key_states)
        streaming_key_states = streaming_key_states[
            :, : self.sink_size + self.recent_size, :, :
        ]

        recent_value_states = streaming_value_states[
            :, -self.recent_size :, :, :
        ].clone()
        streaming_value_states[
            :, self.sink_size : self.sink_size + self.recent_size, :, :
        ].copy_(recent_value_states)
        streaming_value_states = streaming_value_states[
            :, : self.sink_size + self.recent_size, :, :
        ]

    past_key_value = (
        (
            torch.cat([full_key_states, full_value_states], dim=0).transpose(1, 2),
            torch.cat([streaming_key_states, streaming_value_states], dim=0).transpose(
                1, 2
            ),
        )
        if use_cache
        else None
    )

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def mistral_duo_attention_forward_one_way_reordered_static(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    kv_cache: Optional[DuoAttentionStaticKVCache] = None,
    layer_idx: int = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )

    kv_seq_len = q_len
    if kv_cache is not None:
        kv_seq_len += kv_cache.kv_seq_len

    # Replace the Huggingface's apply rotory pos emb with FlashInfer's rope

    # cos, sin = self.rotary_emb(value_states, position_ids)
    # query_states, key_states = apply_rotary_pos_emb(
    #     query_states,
    #     key_states,
    #     cos,
    #     sin,
    #     unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
    # )

    rope_scale = 1.0
    if self.config.rope_scaling is not None:
        rope_scale = self.config.rope_scaling.get("factor", 1.0)
    apply_rope_inplace(
        query_states, key_states, position_ids[:, 0], rope_scale, self.rope_theta
    )

    (
        full_key_states,
        full_value_states,
        streaming_key_states,
        streaming_value_states,
    ) = kv_cache.split_kv(layer_idx, key_states, value_states)
    full_key_states, full_value_states = kv_cache.put_full_kv(
        layer_idx, full_key_states, full_value_states
    )

    if q_len == kv_seq_len:
        # Initial pre-filling
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=0.0,
        )

    else:
        # Decoding or continous filling
        num_full_query_head = (
            kv_cache.num_full_kv_head_list[layer_idx] * self.num_key_value_groups
        )

        (
            cached_streaming_key_states,
            cached_streaming_value_states,
        ) = kv_cache.get_streaming_kv(layer_idx)

        streaming_key_states = torch.cat(
            [cached_streaming_key_states, streaming_key_states], dim=1
        )
        streaming_value_states = torch.cat(
            [cached_streaming_value_states, streaming_value_states], dim=1
        )

        if num_full_query_head > 0:
            full_query_states = query_states[:, :, :num_full_query_head, :]
            full_attn_output = flash_attn_func(
                full_query_states,
                full_key_states,
                full_value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            full_attn_output = None

        if self.num_heads - num_full_query_head > 0:
            streaming_query_states = query_states[:, :, num_full_query_head:, :]
            streaming_attn_output = flash_attn_func(
                streaming_query_states,
                streaming_key_states,
                streaming_value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            streaming_attn_output = None

        if full_attn_output is None:
            attn_output = streaming_attn_output
        elif streaming_attn_output is None:
            attn_output = full_attn_output
        else:
            attn_output = torch.cat([full_attn_output, streaming_attn_output], dim=2)

    kv_cache.compress_and_replace_streaming_kv(
        layer_idx, streaming_key_states, streaming_value_states
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights


def enable_mistral_duo_attention_training(
    model: MistralForCausalLM,
    sink_size,
    recent_size,
    max_length,
    initial_value=1.0,
    enable_ulysses_attention=False,
    streaming_attn_implementation="blocksparse",
):
    enable_tuple_kv_cache_for_mistral(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if streaming_attn_implementation == "blocksparse":
        num_sink_blocks = (sink_size + 127) // 128
        num_recent_blocks = (recent_size + 127) // 128
        num_heads_per_device = model.config.num_attention_heads // int(
            os.environ["WORLD_SIZE"]
        )
        print(
            f"Using blocksparse implementation with {num_sink_blocks} sink blocks, {num_recent_blocks} recent blocks, and {num_heads_per_device} heads per device"
        )
        streaming_mask = generate_streaming_info_blocksparse_flash_attn(
            num_sink_blocks, num_recent_blocks, num_heads_per_device, device
        )
        streaming_attn_func = streaming_attn_blocksparse_flash_attn
    elif streaming_attn_implementation == "sdpa":
        streaming_mask = generate_streaming_mask(
            max_length, sink_size, recent_size, device
        )
        streaming_attn_func = streaming_attn_sdpa
    else:
        raise ValueError(
            f"Unsupported streaming attention implementation: {streaming_attn_implementation}"
        )

    for layer in model.model.layers:
        module = layer.self_attn
        module.forward = types.MethodType(mistral_duo_attention_forward_two_way, module)
        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_parameter(
            "full_attention_heads",
            nn.Parameter(
                torch.ones(
                    module.num_key_value_heads,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                * initial_value
            ),
        )

        module.register_buffer("streaming_mask", streaming_mask)
        if not enable_ulysses_attention:
            module.streaming_attn_func = streaming_attn_func
            module.full_attn_func = flash_attn_func
        else:
            module.streaming_attn_func = UlyssesAttention(
                attn_func=streaming_attn_func,
            )
            module.full_attn_func = UlyssesAttention(
                attn_func=flash_attn_func,
            )


def enable_mistral_duo_attention_eval(
    model: MistralForCausalLM,
    full_attention_heads,
    sink_size,
    recent_size,
):
    enable_tuple_kv_cache_for_mistral(model)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        layer_full_attention_heads = torch.tensor(
            full_attention_heads[idx], device=device, dtype=dtype
        )

        module.forward = types.MethodType(
            mistral_duo_attention_forward_one_way_reordered, module
        )
        module.q_proj = reorder_linear_weights(
            module.q_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "out",
        )
        module.k_proj = reorder_linear_weights(
            module.k_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.v_proj = reorder_linear_weights(
            module.v_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.o_proj = reorder_linear_weights(
            module.o_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "in",
        )
        layer_full_attention_heads = reorder_full_attn_heads(layer_full_attention_heads)

        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_buffer(
            "full_attention_heads",
            layer_full_attention_heads,
        )


def enable_mistral_duo_attention_static_kv_cache_eval(
    model: MistralForCausalLM,
    full_attention_heads,
):
    enable_duo_attention_static_kv_cache_for_mistral(model)
    enable_flashinfer_rmsnorm(model)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        layer_full_attention_heads = torch.tensor(
            full_attention_heads[idx], device=device, dtype=dtype
        )

        module.forward = types.MethodType(
            mistral_duo_attention_forward_one_way_reordered_static, module
        )
        module.q_proj = reorder_linear_weights(
            module.q_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "out",
        )
        module.k_proj = reorder_linear_weights(
            module.k_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.v_proj = reorder_linear_weights(
            module.v_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.o_proj = reorder_linear_weights(
            module.o_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "in",
        )


def get_mistral_full_attention_heads(model):
    full_attention_heads = []
    if isinstance(model, TensorParallelPreTrainedModel):
        for shard in model.wrapped_model.module_shards:
            sharded_full_attention_heads = []
            for layer in shard.model.layers:
                module = layer.self_attn.tp_wrapped_module
                if not hasattr(module, "full_attention_heads"):
                    continue
                sharded_full_attention_heads.append(module.full_attention_heads)
            full_attention_heads.append(sharded_full_attention_heads)
        # concatenate the full_attention_heads from all shards, getting a list of tensors with len = num_layers
        device = full_attention_heads[0][0].device
        full_attention_heads = [
            torch.cat(
                [
                    sharded_heads[layer_idx].to(device)
                    for sharded_heads in full_attention_heads
                ]
            )
            for layer_idx in range(len(full_attention_heads[0]))
        ]
    elif isinstance(model, MistralForCausalLM):
        for layer in model.model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            full_attention_heads.append(module.full_attention_heads)
    elif isinstance(model, MistralModel):
        for layer in model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            full_attention_heads.append(module.full_attention_heads)
    else:
        raise ValueError("Model type not supported")

    return full_attention_heads


def set_mistral_full_attention_heads(model, full_attention_heads):
    if isinstance(model, TensorParallelPreTrainedModel):
        for shard in model.wrapped_model.module_shards:
            for layer_idx, layer in enumerate(shard.model.layers):
                module = layer.self_attn.tp_wrapped_module
                if not hasattr(module, "full_attention_heads"):
                    continue
                module.full_attention_heads.data = full_attention_heads[layer_idx].to(
                    module.full_attention_heads.device,
                    module.full_attention_heads.dtype,
                )
    elif isinstance(model, MistralForCausalLM):
        for layer_idx, layer in enumerate(model.model.layers):
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            module.full_attention_heads.data = full_attention_heads[layer_idx].to(
                module.full_attention_heads.device, module.full_attention_heads.dtype
            )
    elif isinstance(model, MistralModel):
        for layer_idx, layer in enumerate(model.layers):
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            module.full_attention_heads.data = full_attention_heads[layer_idx].to(
                module.full_attention_heads.device, module.full_attention_heads.dtype
            )
    else:
        raise ValueError("Model type not supported")


def map_mistral_full_attention_heads(model, func):
    if isinstance(model, TensorParallelPreTrainedModel):
        for shard in model.wrapped_model.module_shards:
            for layer in shard.model.layers:
                module = layer.self_attn.tp_wrapped_module
                if not hasattr(module, "full_attention_heads"):
                    continue
                func(module.full_attention_heads)
    elif isinstance(model, MistralForCausalLM):
        for layer in model.model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            func(module.full_attention_heads)
    elif isinstance(model, MistralModel):
        for layer in model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            func(module.full_attention_heads)
    else:
        raise ValueError("Model type not supported")
