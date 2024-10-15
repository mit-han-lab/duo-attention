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
from duo_attn.patch import enable_duo_attention_eval
from utils import bench_func


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    tokenizer = get_tokenizer(args.model_name)

    with torch.no_grad():
        model = get_model(args.model_name)

    if model.config.model_type == "mistral":
        model.model._prepare_decoder_attention_mask = lambda *args, **kwargs: None
    elif model.config.model_type == "llama":
        model.model._prepare_decoder_attention_mask = lambda *args, **kwargs: None

    model = to_device(model, args.device)

    if args.attn_load_dir is not None:
        full_attention_heads, sink_size, recent_size = load_attn_pattern(
            args.attn_load_dir
        )

        full_attention_heads, sparsity = sparsify_attention_heads(
            full_attention_heads, args.threshold, args.sparsity
        )
        enable_duo_attention_eval(
            model,
            full_attention_heads,
            16,
            64,
        )

    text = "a\n\n" * args.max_length

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")[
        :, : args.max_length - 1
    ]

    print(input_ids.shape)
    # pre-filling
    torch.cuda.reset_peak_memory_stats()

    def func1():
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
            )

    # ctx_latency, ctx_memory = bench_func(func1, num_steps=20, num_warmup_steps=10)
    ctx_latency, ctx_memory = 0, 0

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True,
        )
    print(
        f"Peak memory usage in the pre-filling stage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB"
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]

    def func2():
        with torch.no_grad():
            _ = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )

    gen_latency, gen_memory = bench_func(func2, num_steps=100, num_warmup_steps=10)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "benchmark_result.txt"), "w") as f:
            print(f"Average generation time: {gen_latency:.2f} ms", file=f)
            print(f"Peak generation memory usage: {gen_memory:.2f} MB", file=f)
            print(f"Average context time: {ctx_latency:.2f} ms", file=f)
            print(f"Peak context memory usage: {ctx_memory:.2f} MB", file=f)
            print(f"Model name: {args.model_name}", file=f)
            print(f"Context length: {args.max_length}", file=f)
            print(f"Sparsity: {sparsity}", file=f)
