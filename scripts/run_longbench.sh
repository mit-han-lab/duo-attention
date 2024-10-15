models="Llama-2-7B-32K-Instruct Llama-3-8B-Instruct-Gradient-1048k"

sparsities="0 0.5 0.75"

tasks="samsum narrativeqa qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"

for model in $models; do
    for task in $tasks; do
        for sparsity in $sparsities; do
            bash scripts/longbench_duo_attn.sh $model $task "attn_patterns/${model}" $sparsity
        done
    done
done

cd eval/LongBench
for model in $model_attn_patterns; do
    model=$(echo $model | cut -d'/' -f1)
    python -u eval.py --model $model &
done
