model_name=$1
attn_pattern=$2
sparsity=$3
max_length=$4

python eval/efficiency/benchmark_static.py \
    --model_name models/${model_name} \
    --max_length $max_length \
    --attn_load_dir $attn_pattern \
    --output_dir outputs/efficiency/${model_name}/gen_ctx=${max_length}_sparsity=${sparsity}_chunk_size=32000 \
    --seed 42 \
    --sparsity $sparsity \
    --prefilling_chunk_size 32000
