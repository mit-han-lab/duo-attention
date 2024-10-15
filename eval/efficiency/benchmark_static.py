import torch
import os

from duo_attn.utils import (
    get_model,
    get_tokenizer,
    parse_args,
    to_device,
    load_attn_pattern,
    seed_everything,
    sparsify_attention_heads,
)
from duo_attn.patch.llama import (
    enable_llama_duo_attention_static_kv_cache_eval,
    DuoAttentionStaticKVCache,
)
from utils import bench_func


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    tokenizer = get_tokenizer(args.model_name)

    with torch.no_grad():
        model = get_model(args.model_name)

    model.eval()

    model = to_device(model, args.device)

    if args.attn_load_dir is not None:
        full_attention_heads, sink_size, recent_size = load_attn_pattern(
            args.attn_load_dir
        )

        full_attention_heads, sparsity = sparsify_attention_heads(
            full_attention_heads, None, args.sparsity
        )
        print(f"True Sparsity: {sparsity}")
        enable_llama_duo_attention_static_kv_cache_eval(model, full_attention_heads)

    text = "a\n\n" * args.max_length

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")[
        :, : args.max_length - 1
    ]

    print(input_ids.shape)

    max_size = input_ids.size(1) + 5
    prefilling_chunk_size = args.prefilling_chunk_size
    print(f"Max size: {max_size}, Prefilling chunk size: {prefilling_chunk_size}")

    kv_cache = DuoAttentionStaticKVCache(
        model,
        full_attention_heads,
        1,
        max_size,
        sink_size,
        recent_size,
    )

    # pre-filling
    def func1():
        with torch.no_grad():
            for i in range(0, input_ids.size(1), prefilling_chunk_size):
                input_chunk = input_ids[:, i : i + prefilling_chunk_size]
                outputs = model(
                    input_ids=input_chunk,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
            kv_cache.clear()

    ctx_latency, ctx_memory = bench_func(func1, num_steps=10, num_warmup_steps=3)

    kv_cache.clear()
    with torch.no_grad():
        for i in range(0, input_ids.size(1), prefilling_chunk_size):
            input_chunk = input_ids[:, i : i + prefilling_chunk_size]
            outputs = model(
                input_ids=input_chunk,
                past_key_values=kv_cache,
                use_cache=True,
            )
    print(
        f"Peak memory usage in the pre-filling stage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB"
    )
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]

    def func2():
        with torch.no_grad():
            _ = model(
                input_ids=pred_token_idx,
                past_key_values=kv_cache,
                use_cache=True,
            )
        kv_cache.evict_last(1)

    gen_latency, gen_memory = bench_func(func2, num_steps=100, num_warmup_steps=50)

    kv_cache_memory_usage = kv_cache.memory_usage / 1024 / 1024
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "benchmark_result.txt"), "w") as f:
            print(f"Average generation time: {gen_latency:.4f} ms", file=f)
            print(f"Peak generation memory usage: {gen_memory:.4f} MB", file=f)
            print(f"Average context time: {ctx_latency:.4f} ms", file=f)
            print(f"Peak context memory usage: {ctx_memory:.4f} MB", file=f)
            print(f"Model name: {args.model_name}", file=f)
            print(f"Context length: {args.max_length}", file=f)
            print(f"Sparsity: {sparsity}", file=f)
            print(f"Prefilling chunk size: {prefilling_chunk_size}", file=f)
            print(f"KV cache memory usage: {kv_cache_memory_usage:.4f} MB", file=f)
