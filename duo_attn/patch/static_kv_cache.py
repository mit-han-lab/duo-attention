import torch

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    CausalLMOutputWithPast,
    Union,
    CrossEntropyLoss,
    BaseModelOutputWithPast,
)
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM,
)

import types
from typing import List, Optional, Tuple, Union


class DuoAttentionStaticKVCache:
    def __init__(
        self,
        model,
        full_attention_heads,
        batch_size,
        max_size,
        sink_size,
        recent_size,
    ):
        self.batch_size = batch_size
        self.max_size = max_size
        self.sink_size = sink_size
        self.recent_size = recent_size

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = model.config.hidden_size // self.num_heads

        self.num_full_kv_head_list = [0] * self.num_layers
        self.num_streaming_kv_head_list = [0] * self.num_layers

        self.kv_seq_len_list = [0] * self.num_layers
        self.streaming_kv_seq_len_list = [0] * self.num_layers

        self.streaming_key_states_list = []
        self.streaming_value_states_list = []
        self.full_key_states_list = []
        self.full_value_states_list = []

        for idx, layer_full_attention_heads in enumerate(full_attention_heads):
            layer_full_attention_heads = torch.tensor(layer_full_attention_heads) > 0.5
            num_full_kv_head = layer_full_attention_heads.sum().item()
            num_streaming_kv_head = self.num_kv_heads - num_full_kv_head

            self.num_full_kv_head_list[idx] = num_full_kv_head
            self.num_streaming_kv_head_list[idx] = num_streaming_kv_head

            streaming_key_states = torch.zeros(
                self.batch_size,
                self.sink_size + self.recent_size,
                num_streaming_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            streaming_value_states = torch.zeros(
                self.batch_size,
                self.sink_size + self.recent_size,
                num_streaming_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            full_key_states = torch.zeros(
                self.batch_size,
                self.max_size,
                num_full_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            full_value_states = torch.zeros(
                self.batch_size,
                self.max_size,
                num_full_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            self.streaming_key_states_list.append(streaming_key_states)
            self.streaming_value_states_list.append(streaming_value_states)
            self.full_key_states_list.append(full_key_states)
            self.full_value_states_list.append(full_value_states)

    @property
    def streaming_kv_seq_len(self):
        return self.streaming_kv_seq_len_list[-1]

    @property
    def kv_seq_len(self):
        return self.kv_seq_len_list[-1]

    def put_full_kv(self, layer_idx, full_key_states, full_value_states):
        incoming_kv_seq_len = full_key_states.shape[1]
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        if incoming_kv_seq_len + kv_seq_len > self.max_size:
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with max size {self.max_size}, current size: {kv_seq_len}."
            )

        self.full_key_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_key_states)
        self.full_value_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_value_states)

        self.kv_seq_len_list[layer_idx] += incoming_kv_seq_len
        return self.get_full_kv(layer_idx)

    def compress_and_replace_streaming_kv(
        self, layer_idx, streaming_key_states, streaming_value_states
    ):
        incoming_kv_seq_len = streaming_key_states.shape[1]
        if incoming_kv_seq_len <= self.sink_size + self.recent_size:
            self.streaming_key_states_list[layer_idx][
                :,
                :incoming_kv_seq_len,
            ].copy_(streaming_key_states)
            self.streaming_value_states_list[layer_idx][
                :,
                :incoming_kv_seq_len,
            ].copy_(streaming_value_states)

            self.streaming_kv_seq_len_list[layer_idx] = incoming_kv_seq_len
        else:
            sink_key_states = streaming_key_states[:, : self.sink_size]
            recent_key_states = streaming_key_states[
                :, incoming_kv_seq_len - self.recent_size : incoming_kv_seq_len
            ]
            self.streaming_key_states_list[layer_idx][:, : self.sink_size].copy_(
                sink_key_states
            )
            self.streaming_key_states_list[layer_idx][
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_key_states)

            sink_value_states = streaming_value_states[:, : self.sink_size]
            recent_value_states = streaming_value_states[
                :, incoming_kv_seq_len - self.recent_size : incoming_kv_seq_len
            ]
            self.streaming_value_states_list[layer_idx][:, : self.sink_size].copy_(
                sink_value_states
            )
            self.streaming_value_states_list[layer_idx][
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_value_states)

            self.streaming_kv_seq_len_list[layer_idx] = (
                self.recent_size + self.sink_size
            )

    def put(self, layer_idx, key_states, value_states):
        incoming_kv_seq_len = key_states.shape[1]
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        if incoming_kv_seq_len + kv_seq_len > self.max_size:
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with max size {self.max_size}, current size: {kv_seq_len}."
            )
        if (
            incoming_kv_seq_len + streaming_kv_seq_len
            > self.sink_size + self.recent_size + self.prefilling_chunk_size
        ):
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with sink size {self.sink_size}, recent size {self.recent_size}, and prefilling chunk size {self.prefilling_chunk_size}, current size: {streaming_kv_seq_len}."
            )

        (
            full_key_states,
            full_value_states,
            streaming_key_states,
            streaming_value_states,
        ) = self.split_kv(layer_idx, key_states, value_states)

        self.full_key_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_key_states)
        self.full_value_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_value_states)

        self.streaming_key_states_list[layer_idx][
            :,
            streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len,
        ].copy_(streaming_key_states)
        self.streaming_value_states_list[layer_idx][
            :,
            streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len,
        ].copy_(streaming_value_states)

        self.update_seq_len(layer_idx, incoming_kv_seq_len)

        return self.get(layer_idx)

    def update_seq_len(self, layer_idx, incoming_kv_seq_len):
        self.kv_seq_len_list[layer_idx] += incoming_kv_seq_len
        self.streaming_kv_seq_len_list[layer_idx] += incoming_kv_seq_len

    def get_full_kv(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        return (
            self.full_key_states_list[layer_idx][:, :kv_seq_len],
            self.full_value_states_list[layer_idx][:, :kv_seq_len],
        )

    def get_streaming_kv(self, layer_idx):
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        return (
            self.streaming_key_states_list[layer_idx][:, :streaming_kv_seq_len],
            self.streaming_value_states_list[layer_idx][:, :streaming_kv_seq_len],
        )

    def get(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        return (
            self.full_key_states_list[layer_idx][:, :kv_seq_len],
            self.full_value_states_list[layer_idx][:, :kv_seq_len],
            self.streaming_key_states_list[layer_idx][:, :streaming_kv_seq_len],
            self.streaming_value_states_list[layer_idx][:, :streaming_kv_seq_len],
        )

    def get_unsliced(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        return (
            kv_seq_len,
            self.full_key_states_list[layer_idx],
            self.full_value_states_list[layer_idx],
            streaming_kv_seq_len,
            self.streaming_key_states_list[layer_idx],
            self.streaming_value_states_list[layer_idx],
        )

    def split_kv(self, layer_idx, key_states, value_states):
        num_full_kv_head = self.num_full_kv_head_list[layer_idx]
        full_key_states = key_states[:, :, :num_full_kv_head, :]
        full_value_states = value_states[:, :, :num_full_kv_head, :]
        streaming_key_states = key_states[:, :, num_full_kv_head:, :]
        streaming_value_states = value_states[:, :, num_full_kv_head:, :]
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
        recent_key_states = self.streaming_key_states_list[layer_idx][
            :, streaming_kv_seq_len - self.recent_size : streaming_kv_seq_len
        ].clone()
        self.streaming_key_states_list[layer_idx][
            :, self.sink_size : self.sink_size + self.recent_size
        ].copy_(recent_key_states)

        recent_value_states = self.streaming_value_states_list[layer_idx][
            :, streaming_kv_seq_len - self.recent_size : streaming_kv_seq_len
        ].clone()
        self.streaming_value_states_list[layer_idx][
            :, self.sink_size : self.sink_size + self.recent_size
        ].copy_(recent_value_states)

        self.streaming_kv_seq_len_list[layer_idx] = self.recent_size + self.sink_size

    def clear(self):
        for layer_idx in range(self.num_layers):
            self.kv_seq_len_list[layer_idx] = 0
            self.streaming_kv_seq_len_list[layer_idx] = 0

    def evict_last(self, num_tokens):
        for layer_idx in range(self.num_layers):
            kv_seq_len = self.kv_seq_len_list[layer_idx]
            streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
            self.kv_seq_len_list[layer_idx] = max(0, kv_seq_len - num_tokens)
            self.streaming_kv_seq_len_list[layer_idx] = max(
                0, streaming_kv_seq_len - num_tokens
            )

    @property
    def memory_usage(self):
        memory_usage = 0
        for layer_idx in range(self.num_layers):
            memory_usage += self.full_key_states_list[layer_idx].element_size() * (
                self.full_key_states_list[layer_idx].numel()
            )
            memory_usage += self.full_value_states_list[layer_idx].element_size() * (
                self.full_value_states_list[layer_idx].numel()
            )
            memory_usage += self.streaming_key_states_list[layer_idx].element_size() * (
                self.streaming_key_states_list[layer_idx].numel()
            )
            memory_usage += self.streaming_value_states_list[
                layer_idx
            ].element_size() * (self.streaming_value_states_list[layer_idx].numel())
        return memory_usage


def duo_attn_static_kv_cache_llama_for_causal_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[DuoAttentionStaticKVCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]

    if self.training:
        logits = self.lm_head(hidden_states)
        logits = logits.float()
    else:
        logits = self.lm_head(hidden_states[:, -1:, :])

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def duo_attn_static_kv_cache_llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[DuoAttentionStaticKVCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values.kv_seq_len
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=past_key_values,
            layer_idx=idx,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def duo_attn_static_kv_cache_llama_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    kv_cache: Optional[DuoAttentionStaticKVCache] = None,
    layer_idx: int = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        kv_cache=kv_cache,
        layer_idx=layer_idx,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs


def enable_duo_attention_static_kv_cache_for_llama(model: LlamaForCausalLM):
    model.model._prepare_decoder_attention_mask = lambda *args, **kwargs: None

    model.model.forward = types.MethodType(
        duo_attn_static_kv_cache_llama_model_forward, model.model
    )
    for idx in range(len(model.model.layers)):
        model.model.layers[idx].forward = types.MethodType(
            duo_attn_static_kv_cache_llama_decoder_layer_forward,
            model.model.layers[idx],
        )
    model.forward = types.MethodType(
        duo_attn_static_kv_cache_llama_for_causal_lm_forward, model
    )


def duo_attn_static_kv_cache_mistral_for_causal_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[DuoAttentionStaticKVCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]

    if self.training:
        logits = self.lm_head(hidden_states)
    else:
        logits = self.lm_head(hidden_states[:, -1:, :])

    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def duo_attn_static_kv_cache_mistral_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[DuoAttentionStaticKVCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values.kv_seq_len
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=past_key_values,
            layer_idx=idx,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def duo_attn_static_kv_cache_mistral_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    kv_cache: Optional[DuoAttentionStaticKVCache] = None,
    layer_idx: int = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        kv_cache=kv_cache,
        layer_idx=layer_idx,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs


def enable_duo_attention_static_kv_cache_for_mistral(model: MistralForCausalLM):
    model.model._prepare_decoder_attention_mask = lambda *args, **kwargs: None

    model.model.forward = types.MethodType(
        duo_attn_static_kv_cache_mistral_model_forward, model.model
    )
    for idx in range(len(model.model.layers)):
        model.model.layers[idx].forward = types.MethodType(
            duo_attn_static_kv_cache_mistral_decoder_layer_forward,
            model.model.layers[idx],
        )
    model.forward = types.MethodType(
        duo_attn_static_kv_cache_mistral_for_causal_lm_forward, model
    )
