export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python demo/run_duo_w8a8kv4.py --len 3300000 --sparsity 0.5
