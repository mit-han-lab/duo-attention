model=$1
task=$2
attn_pattern=$3
sparsity=$4
python -u eval/LongBench/pred.py \
    --model $model --task $task \
    --method duo_attn \
    --attn_load_dir ${attn_pattern} \
    --sparsity $sparsity \
    --sink_size 64 \
    --recent_size 256
