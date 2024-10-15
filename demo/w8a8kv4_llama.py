# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }

# Inspired by the following papers:
# @article{touvron2023llama,
#   title={Llama 2: Open foundation and fine-tuned chat models},
#   author={Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and Almahairi, Amjad and Babaei, Yasmine and Bashlykov, Nikolay and Batra, Soumya and Bhargava, Prajjwal and Bhosale, Shruti and others},
#   journal={arXiv preprint arXiv:2307.09288},
#   year={2023}
# }

# @article{touvron2023llama,
#   title={Llama: Open and efficient foundation language models},
#   author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and others},
#   journal={arXiv preprint arXiv:2302.13971},
#   year={2023}
# }


from typing import Dict, List, Optional

import qserve_backend.fused_attention as fused_attention

# import gc
import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from qserve_backend import fused_kernels
from torch import nn
from transformers import LlamaConfig
from duo_attn.patch.utils import (
    reorder_full_attn_heads,
)

import qserve.utils.constants
from qserve.modeling.layers.activation import SiluAndMulQuant
from qserve.modeling.layers.layernorm import RMSNorm, RMSNormGeneral
from qserve.modeling.layers.quantized_linear import W8A8OF16LinearDynamicInputScale
from qserve.modeling.layers.sampler import Sampler
from qserve.sampling_params import SamplingParams
from qserve.utils.input_metadata import InputMetadata
from qserve.utils.quant_config import QServeQuantConfig
from qserve.utils.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)

from duo_attn.patch.flashinfer_utils import apply_rope_inplace
from flash_attn import flash_attn_func

max_seq_len = qserve.utils.constants.max_seq_len


class LlamaMLP(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        hidden_size = args.hidden_size
        intermediate_size = args.intermediate_size
        self.use_int8 = True

        self.gate_up_proj = W8A8OF16LinearDynamicInputScale(
            hidden_size, 2 * intermediate_size, bias=False
        )
        self.down_proj = W8A8OF16LinearDynamicInputScale(
            intermediate_size, hidden_size, bias=False
        )

        self.act_fn = SiluAndMulQuant(act_sum=False)

    def forward(self, input_metadata: InputMetadata):
        activation_buffer = input_metadata.activation_buffer

        # INT8 in, FP16 out
        self.gate_up_proj(
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.gate_up_proj_act_buffer,
        )

        # FP16 in, INT8 out
        self.act_fn(
            activation_buffer.gate_up_proj_act_buffer,
            activation_buffer.quantized_mlp_act_buffer,
            activation_buffer.quantized_scale_buffer,
        )

        self.down_proj(
            activation_buffer.quantized_mlp_act_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.out_down_proj_act_buffer,
        )


class LlamaAttention(nn.Module):
    def __init__(
        self,
        args,
        layer_idx: int,
        kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        hidden_size = args.hidden_size
        num_heads = args.num_attention_heads
        num_kv_heads = args.num_key_value_heads
        rope_theta = getattr(args, "rope_theta", 10000)
        rope_scaling = getattr(args, "rope_scaling", None)
        max_position_embeddings = args.max_position_embeddings

        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        num_kv_heads_replicas = max(1, tp_size // self.total_num_kv_heads)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_int8 = True

        # if kv_cache_config is None:
        #     self.kv_cache_config = {"INT4_ENABLED": False, "ZEROS_ENABLED": False}
        #     print("[Warning] kv cache config is not provided, using default config KV8")
        # else:
        #     self.kv_cache_config = kv_cache_config

        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False
        self.qkv_proj = W8A8OF16LinearDynamicInputScale(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads * num_kv_heads_replicas)
            * self.head_dim,
            bias=attention_bias,
        )

        self.o_proj = W8A8OF16LinearDynamicInputScale(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
        )

        self.kv_max_seq_len = min(max_seq_len, self.max_position_embeddings)

        self.invoke_quant = self.invoke_quant_wo_act_sum

    def invoke_quant_wo_act_sum(self, activation_buffer, attn_output):
        fused_kernels.invoke_quant(
            activation_buffer.quantized_hidden_states_buffer,
            attn_output,
            activation_buffer.quantized_scale_buffer,
        )

    def forward(
        self,
        input_metadata: InputMetadata,
        kv_cache,
    ):
        activation_buffer = input_metadata.activation_buffer
        # INT8 in, FP16 out for this module
        # print(self.layer_idx, "begin", hidden_states.isnan().sum(), input_scale.shape)
        self.qkv_proj(
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.qkv_proj_act_buffer,
        )

        query_states, key_states, value_states = (
            activation_buffer.qkv_proj_act_buffer.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
        )

        q_len = activation_buffer.batched_seq_len
        bsz = query_states.size(0) // q_len

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        kv_seq_len = key_states.shape[1]
        if kv_cache is not None:
            kv_seq_len += kv_cache.kv_seq_len

        rope_scale = 1.0
        apply_rope_inplace(
            query_states,
            key_states,
            kv_cache.position_ids_offset,
            rope_scale,
            self.rope_theta,
            kv_cache.indptr,
        )

        num_full_query_head = (
            kv_cache.num_full_kv_head_list[self.layer_idx]
            * self.num_heads
            // self.num_kv_heads
        )
        num_full_kv_head = kv_cache.num_full_kv_head_list[self.layer_idx]

        (
            full_key_states,
            full_value_states,
            streaming_key_states,
            streaming_value_states,
        ) = kv_cache.put(self.layer_idx, key_states, value_states)

        if q_len == kv_seq_len:
            # pre-filling: use flash attention
            full_key_states = key_states[:, :, :num_full_kv_head, :]
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            # decoding or continous filling
            if full_key_states.numel() > 0:
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

            if streaming_key_states.numel() > 0:
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
                attn_output = torch.cat(
                    [full_attn_output, streaming_attn_output], dim=2
                )

        attn_output = attn_output.reshape(bsz * q_len, self.hidden_size)

        kv_cache.compress(self.layer_idx)

        # FP16 in, INT8 out
        self.invoke_quant(activation_buffer, attn_output)
        # INT8 in, FP16 out
        self.o_proj(
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.out_down_proj_act_buffer,
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_int8 = True
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = LlamaAttention(
            config,
            layer_idx=layer_idx,
            kv_cache_config=kv_cache_config,
        )
        self.mlp = LlamaMLP(config)

        self.input_layernorm = RMSNormGeneral(
            config.hidden_size,
            act_sum=False,
            eps=config.rms_norm_eps,
            use_per_token_quant=True,
        )
        self.post_attention_layernorm = RMSNormGeneral(
            config.hidden_size,
            act_sum=False,
            eps=config.rms_norm_eps,
            use_per_token_quant=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        kv_cache,
    ) -> torch.Tensor:
        # FP16 in FP16 out
        activation_buffer = input_metadata.activation_buffer
        # Self Attention
        residual = hidden_states
        # INT8 quantization
        self.input_layernorm(
            hidden_states,
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
        )
        # INT8 -> FP16
        self.self_attn(input_metadata, kv_cache)
        residual += activation_buffer.out_down_proj_act_buffer
        # Fully Connected
        # residual = hidden_states
        # FP16 -> INT8
        self.post_attention_layernorm(
            hidden_states,
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
        )
        # INT8 -> FP16
        self.mlp(input_metadata)
        residual += activation_buffer.out_down_proj_act_buffer
        return residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_kv_cache: bool = True,
        kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                (
                    LlamaDecoderLayer(config, i, kv_cache_config)
                    if quant_kv_cache
                    else None
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_metadata: InputMetadata,
        kv_cache,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if inputs_embeds is None:  # For VLM models
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = inputs_embeds
            q_len = hidden_states.size(1)
            kv_cache.position_ids_offset.fill_(kv_cache.kv_seq_len)
            kv_cache.indptr[1] = q_len
            for i in range(len(self.layers)):
                layer = self.layers[i]
                hidden_states = layer(hidden_states, input_metadata, kv_cache)
            hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        sampling_params: SamplingParams,
        quant_config: Optional[QServeQuantConfig] = QServeQuantConfig(weight_bits=8),
        kv_cache_config: Optional[Dict] = None,
        quant_path: Optional[str] = None,
    ) -> None:
        quant_kv_cache = True

        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = LlamaModel(config, quant_kv_cache, kv_cache_config=kv_cache_config)
        vocab_size = config.vocab_size
        # NOTE: The LM head is not quantized.
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self._column_parallel_layers = []
        self._row_parallel_layers = ["o_proj", "down_proj"]
        self.sampler = Sampler(sampling_params)

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        self.hidden_size = hidden_size
        tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if quant_path is not None:
            self.load_weights(quant_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_metadata = InputMetadata(
            is_prompt=True,
            context_lens=[input_ids.shape[1]],
            padding_offsets=None,
            cu_seqlens=None,
            max_seq_len=0,
            max_block_table_len=None,
            block_tables=None,
            kv_cache_dtype=None,
            kv_scales=None,
            batched_seq_len=input_ids.shape[1],
            model=self,
        )

        hidden_states = self.model(input_ids, input_metadata, kv_cache, inputs_embeds)
        if input_metadata.is_prompt:
            output = self.lm_head(
                hidden_states[
                    :, input_metadata.activation_buffer.batched_seq_len - 1 :, :
                ]
            )  # only compute last logits
        else:
            output = self.lm_head(hidden_states)
        return output  # .float()

    def sample(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        input_metadata: InputMetadata,
    ):
        return self.sampler(input_ids, logits, input_metadata)

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        if self.quant_config is None:
            col_weight_suffixes = ["weight"]
            row_weight_suffixes = ["weight"]
        else:
            col_weight_suffixes = self.quant_config.get_col_parallel_tensor_names()
            row_weight_suffixes = self.quant_config.get_row_parallel_tensor_names()

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in col_weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in row_weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        # TODO fix the tp parallelism
        # tp_size = get_tensor_model_parallel_world_size()
        # tp_rank = get_tensor_model_parallel_rank()
        tp_size = 1
        tp_rank = 0

        q_proj_shard_size = self.config.hidden_size // tp_size
        num_kv_heads_replicas = max(1, tp_size // self.config.num_key_value_heads)
        num_kv_heads_per_gpu = max(1, self.config.num_key_value_heads // tp_size)
        kv_proj_shard_size = (
            self.config.hidden_size
            // self.config.num_attention_heads
            * num_kv_heads_per_gpu
        )
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue
            # bias is useless for llama
            if "bias" in name:
                continue

            packed_dim = None
            is_transposed = False
            if self.quant_config is not None:
                packed_dim = self.quant_config.get_packed_dim(name)
                is_transposed = self.quant_config.is_transposed(name)
            if is_transposed:
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                loaded_weight = loaded_weight.T

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                # print(weight_name)
                param = state_dict[name.replace(weight_name, "qkv_proj")]
                if is_transposed:
                    param = param.T

                if packed_dim is not None:
                    shard_dim = 0 if not is_transposed else 1
                    if packed_dim == shard_dim:
                        shard_size //= self.quant_config.pack_factor
                        offset //= self.quant_config.pack_factor

                if weight_name in ["k_proj", "v_proj"]:
                    shard_id = tp_rank // num_kv_heads_replicas
                else:
                    shard_id = tp_rank
                loaded_weight = loaded_weight[
                    shard_size * shard_id : shard_size * (shard_id + 1)
                ]
                param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight.shape

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                if is_transposed:
                    param = param.T

                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                param_slice = param.data[
                    shard_size * stride_id : shard_size * (stride_id + 1)
                ]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            if is_transposed:
                param = param.T

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tp_rank,
            )


@torch.no_grad()
def reorder_linear_weights(
    weight,
    full_attention_heads: torch.Tensor,
    repeat_num,
    reorder_channel,
    dequant_scale=None,
):
    assert reorder_channel in ["in", "out"]
    full_attention_heads = torch.repeat_interleave(
        full_attention_heads, repeats=repeat_num
    ).to(weight.device)
    full_attn_mask = full_attention_heads > 0.5
    if reorder_channel == "in":
        weight1 = weight[:, full_attn_mask]
        weight2 = weight[:, ~full_attn_mask]
        reordered_weight = torch.cat([weight1, weight2], dim=1)
    else:
        weight1 = weight[full_attn_mask, :]
        weight2 = weight[~full_attn_mask, :]
        reordered_weight = torch.cat([weight1, weight2], dim=0)

        dequant_scale1 = dequant_scale[full_attn_mask]
        dequant_scale2 = dequant_scale[~full_attn_mask]
        dequant_scale = torch.cat([dequant_scale1, dequant_scale2], dim=0)
    weight = reordered_weight
    return weight, dequant_scale


def enable_llama_duo_attention_eval(
    model: LlamaForCausalLM,
    full_attention_heads,
    sink_size,
    recent_size,
):

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        layer_full_attention_heads = torch.tensor(
            full_attention_heads[idx], device=device, dtype=dtype
        )

        (
            module.qkv_proj.weight.data[: module.q_size, :],
            module.qkv_proj.dequant_scale[: module.q_size],
        ) = reorder_linear_weights(
            module.qkv_proj.weight.data[: module.q_size, :],
            layer_full_attention_heads,
            module.num_heads // module.num_kv_heads * module.head_dim,
            "out",
            module.qkv_proj.dequant_scale[: module.q_size],
        )

        (
            module.qkv_proj.weight.data[
                module.q_size : module.q_size + module.kv_size, :
            ],
            module.qkv_proj.dequant_scale[
                module.q_size : module.q_size + module.kv_size
            ],
        ) = reorder_linear_weights(
            module.qkv_proj.weight.data[
                module.q_size : module.q_size + module.kv_size, :
            ],
            layer_full_attention_heads,
            module.head_dim,
            "out",
            module.qkv_proj.dequant_scale[
                module.q_size : module.q_size + module.kv_size
            ],
        )

        (
            module.qkv_proj.weight.data[module.q_size + module.kv_size :, :],
            module.qkv_proj.dequant_scale[module.q_size + module.kv_size :],
        ) = reorder_linear_weights(
            module.qkv_proj.weight.data[module.q_size + module.kv_size :, :],
            layer_full_attention_heads,
            module.head_dim,
            "out",
            module.qkv_proj.dequant_scale[module.q_size + module.kv_size :],
        )

        module.o_proj.weight.data, _ = reorder_linear_weights(
            module.o_proj.weight.data,
            layer_full_attention_heads,
            module.num_heads // module.num_kv_heads * module.head_dim,
            "in",
        )

        layer_full_attention_heads = reorder_full_attn_heads(layer_full_attention_heads)

        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_buffer(
            "full_attention_heads",
            layer_full_attention_heads,
        )
