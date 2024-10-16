model_name="Llama-3-8B-Instruct-Gradient-1048k"
attn_pattern_name="lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
sparsities="0 0.5"
max_lengths="100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000"

for max_length in $max_lengths; do
    for sparsity in $sparsities; do
        CUDA_VISIBLE_DEVICES=0 bash scripts/efficiency.sh $model_name "attn_patterns/${model_name}/${attn_pattern_name}" $sparsity $max_length
    done
done

model_name="Llama-2-7B-32K-Instruct"
attn_pattern_name="lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
sparsities="0 0.75"
max_lengths="20000 40000 60000 80000 100000 120000 140000 160000 180000 200000"

for max_length in $max_lengths; do
    for sparsity in $sparsities; do
        CUDA_VISIBLE_DEVICES=0 bash scripts/efficiency.sh $model_name "attn_patterns/${model_name}/${attn_pattern_name}" $sparsity $max_length
    done
done
