import torch


class QuantizedCache:
    def __init__(
        self, batch_size, max_size, num_kv_heads, head_dim, device, group_size
    ):
        self.batch_size = batch_size
        self.max_size = max_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.group_size = group_size

        self.num_groups = head_dim // group_size
        quantized_dim = head_dim // 2

        self.quantized_data = torch.empty(
            batch_size,
            max_size,
            num_kv_heads,
            quantized_dim,
            device=device,
            dtype=torch.uint8,
        )

        self.scale = torch.empty(
            batch_size,
            max_size,
            num_kv_heads,
            self.num_groups,
            device=device,
            dtype=torch.float16,
        )

        self.zero_point = torch.empty(
            batch_size,
            max_size,
            num_kv_heads,
            self.num_groups,
            device=device,
            dtype=torch.float16,
        )


from torch.utils.cpp_extension import load

# Compile and load the CUDA kernel
module = load(
    name="quantize_int4",
    sources=[
        "demo/quantize_int4.cu"
    ],  # Save the CUDA kernel code in 'quantize_int4.cu'
    extra_cuda_cflags=["--use_fast_math"],
    verbose=False,
)


def quantize_int4_with_zero_point_per_group(
    q_packed, scale, zero_point, tensor, group_size
):
    if tensor.numel() == 0:
        return q_packed[:0], scale[:0], zero_point[:0]

    batch_size, seq_len, num_heads, head_dim = tensor.shape
    num_groups = head_dim // group_size
    packed_group_size = group_size // 2
    total_packed_size = num_groups * packed_group_size

    q_packed = q_packed[: batch_size * seq_len * num_heads * total_packed_size].view(
        batch_size, seq_len, num_heads, total_packed_size
    )
    scale = scale[: batch_size * seq_len * num_heads * num_groups].view(
        batch_size, seq_len, num_heads, num_groups
    )
    zero_point = zero_point[: batch_size * seq_len * num_heads * num_groups].view(
        batch_size, seq_len, num_heads, num_groups
    )

    module.quantize_int4_with_zero_point_per_group(
        tensor,
        q_packed,
        scale,
        zero_point,
        group_size,
    )

    return q_packed, scale, zero_point


def dequantize_int4_with_zero_point_per_group(
    q_packed, scale, zero_point, head_dim, group_size, buffer
):
    if q_packed.numel() == 0:
        return buffer[:0]
        # return torch.empty(0, dtype=torch.float16, device=q_packed.device)

    batch_size, seq_len, num_heads, _ = q_packed.shape

    group_size_packed = group_size // 2
    q_packed = q_packed.view(-1, group_size_packed)
    N = q_packed.shape[0]

    output = buffer[: N * group_size].view(N, group_size)

    module.dequantize_int4_with_zero_point_per_group(
        q_packed, scale, zero_point, group_size, buffer, N
    )

    output = output.view(batch_size, seq_len, num_heads, head_dim)

    return output


class DuoAttentionStaticINT4KVCache:
    def __init__(
        self,
        model,
        full_attention_heads,
        batch_size,
        max_size,
        sink_size,
        recent_size,
        prefilling_chunk_size,
    ):
        self.batch_size = batch_size
        self.max_size = max_size
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.prefilling_chunk_size = prefilling_chunk_size

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = model.config.hidden_size // self.num_heads

        self.group_size = 128

        self.num_full_kv_head_list = [0] * self.num_layers
        self.num_streaming_kv_head_list = [0] * self.num_layers

        self.streaming_key_caches = []
        self.streaming_value_caches = []
        self.full_key_caches = []
        self.full_value_caches = []

        max_num_full_kv_head = 0
        max_num_streaming_kv_head = 0
        for idx, layer_full_attention_heads in enumerate(full_attention_heads):
            layer_full_attention_heads = torch.tensor(layer_full_attention_heads) > 0.5
            num_full_kv_head = layer_full_attention_heads.sum().item()
            num_streaming_kv_head = self.num_kv_heads - num_full_kv_head

            max_num_full_kv_head = max(max_num_full_kv_head, num_full_kv_head)
            max_num_streaming_kv_head = max(
                max_num_streaming_kv_head, num_streaming_kv_head
            )

            self.num_full_kv_head_list[idx] = num_full_kv_head
            self.num_streaming_kv_head_list[idx] = num_streaming_kv_head

            streaming_key_cache = QuantizedCache(
                self.batch_size,
                self.sink_size + self.recent_size + self.prefilling_chunk_size,
                num_streaming_kv_head,
                self.head_dim,
                device=self.device,
                group_size=self.group_size,
            )

            streaming_value_cache = QuantizedCache(
                self.batch_size,
                self.sink_size + self.recent_size + self.prefilling_chunk_size,
                num_streaming_kv_head,
                self.head_dim,
                device=self.device,
                group_size=self.group_size,
            )

            full_key_cache = QuantizedCache(
                self.batch_size,
                self.max_size,
                num_full_kv_head,
                self.head_dim,
                device=self.device,
                group_size=self.group_size,
            )

            full_value_cache = QuantizedCache(
                self.batch_size,
                self.max_size,
                num_full_kv_head,
                device=self.device,
                head_dim=self.head_dim,
                group_size=self.group_size,
            )

            self.streaming_key_caches.append(streaming_key_cache)
            self.streaming_value_caches.append(streaming_value_cache)
            self.full_key_caches.append(full_key_cache)
            self.full_value_caches.append(full_value_cache)

        self.kv_seq_len_list = [0] * self.num_layers
        self.streaming_kv_seq_len_list = [0] * self.num_layers

        max_packed_num_full = self.max_size * max_num_full_kv_head * self.head_dim
        self.fp16_full_key_buffer = torch.empty(
            (max_packed_num_full,), device=self.device, dtype=torch.float16
        )
        self.fp16_full_value_buffer = torch.empty(
            (max_packed_num_full,), device=self.device, dtype=torch.float16
        )

        max_packed_num_streaming = (
            (self.sink_size + self.recent_size + self.prefilling_chunk_size)
            * max_num_streaming_kv_head
            * self.head_dim
        )
        self.fp16_streaming_key_buffer = torch.empty(
            (max_packed_num_streaming,), device=self.device, dtype=torch.float16
        )
        self.fp16_streaming_value_buffer = torch.empty(
            (max_packed_num_streaming,), device=self.device, dtype=torch.float16
        )

        max_packed_num_full = (
            self.prefilling_chunk_size * max_num_full_kv_head * self.head_dim
        )
        self.q_packed_buffer = torch.empty(
            (max_packed_num_full // 2,), device=self.device, dtype=torch.uint8
        )
        self.scale_buffer = torch.empty(
            (max_packed_num_full // self.group_size,),
            device=self.device,
            dtype=torch.float16,
        )
        self.zero_point_buffer = torch.empty(
            (max_packed_num_full // self.group_size,),
            device=self.device,
            dtype=torch.float16,
        )

        self.position_ids_offset = torch.empty(
            (self.batch_size,), device=self.device, dtype=torch.long
        )
        self.indptr = torch.zeros(
            (self.batch_size + 1,), dtype=torch.int32, device=self.device
        )

    @property
    def streaming_kv_seq_len(self):
        return self.streaming_kv_seq_len_list[-1]

    @property
    def kv_seq_len(self):
        return self.kv_seq_len_list[-1]

    def put(self, layer_idx, key_states, value_states):
        num_full_kv_head = self.num_full_kv_head_list[layer_idx]
        num_streaming_kv_head = self.num_streaming_kv_head_list[layer_idx]

        incoming_kv_seq_len = key_states.shape[1]
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        if incoming_kv_seq_len + kv_seq_len > self.max_size:
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with max size {self.max_size}, current size: {kv_seq_len}."
            )

        full_key_states = key_states[:, :, :num_full_kv_head, :]
        full_value_states = value_states[:, :, :num_full_kv_head, :]

        streaming_key_states = key_states[:, :, num_full_kv_head:, :]
        streaming_value_states = value_states[:, :, num_full_kv_head:, :]

        full_key_cache = self.full_key_caches[layer_idx]
        full_value_cache = self.full_value_caches[layer_idx]
        streaming_key_cache = self.streaming_key_caches[layer_idx]
        streaming_value_cache = self.streaming_value_caches[layer_idx]

        if num_full_kv_head > 0:
            q_full_key_states, scale_full_key, zero_point_full_key = (
                quantize_int4_with_zero_point_per_group(
                    self.q_packed_buffer,
                    self.scale_buffer,
                    self.zero_point_buffer,
                    full_key_states,
                    self.group_size,
                )
            )

            full_key_cache.quantized_data[
                :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
            ].copy_(q_full_key_states)
            full_key_cache.scale[
                :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
            ].copy_(scale_full_key)
            full_key_cache.zero_point[
                :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
            ].copy_(zero_point_full_key)

            q_full_value_states, scale_full_value, zero_point_full_value = (
                quantize_int4_with_zero_point_per_group(
                    self.q_packed_buffer,
                    self.scale_buffer,
                    self.zero_point_buffer,
                    full_value_states,
                    self.group_size,
                )
            )

            full_value_cache.quantized_data[
                :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
            ].copy_(q_full_value_states)
            full_value_cache.scale[
                :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
            ].copy_(scale_full_value)
            full_value_cache.zero_point[
                :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
            ].copy_(zero_point_full_value)

        if num_streaming_kv_head > 0:
            q_streaming_key_states, scale_streaming_key, zero_point_streaming_key = (
                quantize_int4_with_zero_point_per_group(
                    self.q_packed_buffer,
                    self.scale_buffer,
                    self.zero_point_buffer,
                    streaming_key_states,
                    self.group_size,
                )
            )

            streaming_key_cache.quantized_data[
                :, streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len
            ].copy_(q_streaming_key_states)
            streaming_key_cache.scale[
                :, streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len
            ].copy_(scale_streaming_key)
            streaming_key_cache.zero_point[
                :, streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len
            ].copy_(zero_point_streaming_key)

            (
                q_streaming_value_states,
                scale_streaming_value,
                zero_point_streaming_value,
            ) = quantize_int4_with_zero_point_per_group(
                self.q_packed_buffer,
                self.scale_buffer,
                self.zero_point_buffer,
                streaming_value_states,
                self.group_size,
            )

            streaming_value_cache.quantized_data[
                :, streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len
            ].copy_(q_streaming_value_states)
            streaming_value_cache.scale[
                :, streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len
            ].copy_(scale_streaming_value)
            streaming_value_cache.zero_point[
                :, streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len
            ].copy_(zero_point_streaming_value)

        self.kv_seq_len_list[layer_idx] += incoming_kv_seq_len
        self.streaming_kv_seq_len_list[layer_idx] += incoming_kv_seq_len

        return self.get(layer_idx)

    def get(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]

        full_key_cache = self.full_key_caches[layer_idx]
        full_value_cache = self.full_value_caches[layer_idx]
        streaming_key_cache = self.streaming_key_caches[layer_idx]
        streaming_value_cache = self.streaming_value_caches[layer_idx]

        num_full_kv_head = self.num_full_kv_head_list[layer_idx]
        num_streaming_kv_head = self.num_streaming_kv_head_list[layer_idx]

        if num_full_kv_head > 0:
            full_key_states = dequantize_int4_with_zero_point_per_group(
                full_key_cache.quantized_data[:, :kv_seq_len],
                full_key_cache.scale[:, :kv_seq_len],
                full_key_cache.zero_point[:, :kv_seq_len],
                head_dim=full_key_cache.head_dim,
                group_size=full_key_cache.group_size,
                buffer=self.fp16_full_key_buffer,
            )
            full_value_states = dequantize_int4_with_zero_point_per_group(
                full_value_cache.quantized_data[:, :kv_seq_len],
                full_value_cache.scale[:, :kv_seq_len],
                full_value_cache.zero_point[:, :kv_seq_len],
                head_dim=full_value_cache.head_dim,
                group_size=full_value_cache.group_size,
                buffer=self.fp16_full_value_buffer,
            )
        else:
            full_key_states = torch.empty(0, device=self.device, dtype=torch.float16)
            full_value_states = torch.empty(0, device=self.device, dtype=torch.float16)

        if num_streaming_kv_head > 0:
            streaming_key_states = dequantize_int4_with_zero_point_per_group(
                streaming_key_cache.quantized_data[:, :streaming_kv_seq_len],
                streaming_key_cache.scale[:, :streaming_kv_seq_len],
                streaming_key_cache.zero_point[:, :streaming_kv_seq_len],
                head_dim=streaming_key_cache.head_dim,
                group_size=streaming_key_cache.group_size,
                buffer=self.fp16_streaming_key_buffer,
            )
            streaming_value_states = dequantize_int4_with_zero_point_per_group(
                streaming_value_cache.quantized_data[:, :streaming_kv_seq_len],
                streaming_value_cache.scale[:, :streaming_kv_seq_len],
                streaming_value_cache.zero_point[:, :streaming_kv_seq_len],
                head_dim=streaming_value_cache.head_dim,
                group_size=streaming_value_cache.group_size,
                buffer=self.fp16_streaming_value_buffer,
            )
        else:
            streaming_key_states = torch.empty(
                0, device=self.device, dtype=torch.float16
            )
            streaming_value_states = torch.empty(
                0, device=self.device, dtype=torch.float16
            )

        return (
            full_key_states,
            full_value_states,
            streaming_key_states,
            streaming_value_states,
        )

    def compress(self, layer_idx):
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        if streaming_kv_seq_len <= self.recent_size + self.sink_size:
            return

        streaming_key_cache = self.streaming_key_caches[layer_idx]
        streaming_value_cache = self.streaming_value_caches[layer_idx]

        num_streaming_kv_head = self.num_streaming_kv_head_list[layer_idx]

        if num_streaming_kv_head > 0:
            start_idx = streaming_kv_seq_len - self.recent_size
            end_idx = streaming_kv_seq_len

            recent_quantized_data_key = streaming_key_cache.quantized_data[
                :, start_idx:end_idx
            ].clone()
            recent_scale_key = streaming_key_cache.scale[:, start_idx:end_idx].clone()
            recent_zero_point_key = streaming_key_cache.zero_point[
                :, start_idx:end_idx
            ].clone()

            streaming_key_cache.quantized_data[
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_quantized_data_key)
            streaming_key_cache.scale[
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_scale_key)
            streaming_key_cache.zero_point[
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_zero_point_key)

            recent_quantized_data_value = streaming_value_cache.quantized_data[
                :, start_idx:end_idx
            ].clone()
            recent_scale_value = streaming_value_cache.scale[
                :, start_idx:end_idx
            ].clone()
            recent_zero_point_value = streaming_value_cache.zero_point[
                :, start_idx:end_idx
            ].clone()

            streaming_value_cache.quantized_data[
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_quantized_data_value)
            streaming_value_cache.scale[
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_scale_value)
            streaming_value_cache.zero_point[
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_zero_point_value)

            self.streaming_kv_seq_len_list[layer_idx] = (
                self.recent_size + self.sink_size
            )
