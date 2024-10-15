import torch
from tqdm import tqdm

import torch.backends.cudnn as cudnn


def bench_func(func, num_steps=100, num_warmup_steps=5):
    cudnn.benchmark = True
    pbar = tqdm(range(num_warmup_steps), desc="Warming up...")
    for _ in pbar:
        func()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pbar = tqdm(range(num_steps), desc="Benchmarking Latency and Memory...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in pbar:
        func()
    end.record()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)
    avg_time = total_time / num_steps
    print(f"Average latency: {avg_time:.2f} ms")
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak memory usage: {peak_memory:.2f} MB")
    return (
        avg_time,
        peak_memory,
    )
