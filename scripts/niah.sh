cd eval/needle
mkdir -p logs img results

attn_pattern=$1
sparsity=$2
model=$3
context_lengths_min=$4
s_len=$5
pretrained_len=$6
model_provider=$7

# cut the last part of the path of the attn_pattern to get the name
attn_pattern_name=$(echo $attn_pattern | rev | cut -d'/' -f1 | rev)

suffix="duo_attn-attn_pattern=${attn_pattern_name}-sparsity=${sparsity}"
(
    python -u needle_in_haystack.py --s_len $s_len \
        --e_len $pretrained_len \
        --context_lengths_min $context_lengths_min \
        --context_lengths_max $pretrained_len \
        --model_provider $model_provider \
        --model_name_suffix $suffix \
        --attn_load_dir ../../$attn_pattern \
        --sparsity $sparsity \
        --simulation_length 0 \
        --context_lengths_num_intervals 13 \
        --document_depth_percent_intervals 10 \
        --sink_size 64 \
        --recent_size 256 \
        --prefilling_chunk_size 32000 \
        --model_path ../../models/${model}
) 2>&1 | tee logs/eval_${model}_${suffix}.log

python visualize.py \
    --folder_path "results/${model}_${suffix}/" \
    --model_name "${model} DuoAttention Sparsity=${sparsity}" \
    --pretrained_len $pretrained_len
