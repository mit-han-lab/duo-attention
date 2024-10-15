import torch

try:
    import xformers.ops as xops
except ImportError:
    xops = None

try:
    from block_sparse_attn import block_streaming_attn_func
except ImportError:
    block_streaming_attn_func = None


@torch.no_grad()
def generate_streaming_mask(seq_len, sink_size, recent_size, device):
    # round seq_len to the nearest multiple of 8
    seq_len = (seq_len + 7) // 8 * 8
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool)
    causal_mask = ~torch.triu(ones, diagonal=1)
    recent_mask = torch.triu(ones, diagonal=-recent_size + 1)
    sink_mask = ones
    sink_mask[:, sink_size:] = False
    mask = (recent_mask | sink_mask) & causal_mask
    return mask.to(device=device).unsqueeze(0).unsqueeze(0)


def streaming_attn_sdpa(query_states, key_states, value_states, streaming_causal_mask):
    bsz, seq_len, num_heads, head_dim = query_states.size()

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    streaming_attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=streaming_causal_mask[:, :, :seq_len, :seq_len],
        dropout_p=0.0,
        enable_gqa=True,
    )

    return streaming_attn_output.transpose(1, 2)


def streaming_attn_xformers(
    query_states, key_states, value_states, streaming_causal_mask
):
    # query_states: [bsz, seq_len, num_heads, head_dim]
    # key_states: [bsz, seq_len, num_heads, head_dim]
    # value_states: [bsz, seq_len, num_heads, head_dim]
    # Return: [bsz, seq_len, num_heads, head_dim]

    bsz, seq_len, num_heads, head_dim = query_states.size()
    attn_bias = streaming_causal_mask[:, :, :seq_len, :seq_len].expand(
        bsz, num_heads, seq_len, seq_len
    )

    streaming_attn_output = xops.memory_efficient_attention(
        query_states,
        key_states,
        value_states,
        attn_bias=attn_bias,
        p=0.0,
    )

    return streaming_attn_output


def generate_streaming_info_blocksparse_flash_attn(
    sink_block_num, local_block_num, n_query_heads, device
):
    streaming_info = torch.tensor(
        [sink_block_num, local_block_num] * n_query_heads,
        device=device,
        dtype=torch.int32,
    )
    return streaming_info


def streaming_attn_blocksparse_flash_attn(
    query_states, key_states, value_states, streaming_info
):
    bts, seqlen, query_heads, head_dim = query_states.size()
    key_value_heads = key_states.size(2)
    query_unpad = query_states.view(bts * seqlen, query_heads, head_dim)
    key_unpad = key_states.view(bts * seqlen, key_value_heads, head_dim)
    value_unpad = value_states.view(bts * seqlen, key_value_heads, head_dim)
    cu_seqlens = torch.arange(
        0, (bts + 1) * seqlen, step=seqlen, dtype=torch.int32, device=query_unpad.device
    )
    head_mask_type = torch.tensor(
        [-1] * query_heads, device=query_unpad.device, dtype=torch.int32
    )
    attn_output = block_streaming_attn_func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens,
        cu_seqlens,
        head_mask_type,
        streaming_info,
        seqlen,
        seqlen,
        p_dropout=0.0,
        is_causal=True,
    )
    return attn_output.reshape(bts, seqlen, query_heads, head_dim)
