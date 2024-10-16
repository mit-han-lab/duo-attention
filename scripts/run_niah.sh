model="Llama-3-8B-Instruct-Gradient-1048k"
model_provider=LLaMA
context_lengths_min=80000
pretrained_len=1048000
sparsity=0.5
attn_pattern="attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/niah.sh $attn_pattern $sparsity $model $context_lengths_min 0 $pretrained_len $model_provider

model="Llama-2-7B-32K-Instruct"
model_provider=LLaMA
context_lengths_min=2000
pretrained_len=32000
sparsity=0.75
attn_pattern="attn_patterns/Llama-2-7B-32K-Instruct/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"

CUDA_VISIBLE_DEVICES=0 bash scripts/niah.sh $attn_pattern $sparsity $model $context_lengths_min 0 $pretrained_len $model_provider
