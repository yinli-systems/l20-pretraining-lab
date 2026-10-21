#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=x eval.sh <model_path_or_id> <name> [chat|base]
set -u
source /workspace/l20-posttrain/.venv/bin/activate
M=$1; N=$2; MODE=${3:-chat}
O=/workspace/l20-posttrain/eval/$N; mkdir -p $O
export HF_ALLOW_CODE_EVAL=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
VA="pretrained=$M,dtype=bfloat16,gpu_memory_utilization=0.8,max_model_len=2048,enable_prefix_caching=True"
CHAT=""; [ "$MODE" = chat ] && CHAT="--apply_chat_template --fewshot_as_multiturn"
run() { lm_eval --model vllm --model_args "$VA" --batch_size auto --seed 0 --output_path $O "$@" 2>&1 | grep -aE "^\||Error:|error:" ; }
# zero-shot chat math with \boxed{} answers (math-verify)
[ "$MODE" = chat ] && python /workspace/l20-posttrain/math_eval.py $M $O/mathchat.json 2>&1 | grep -aE "^(math500|gsm8k) |Error"
# generative (chat template for chat models)
run --tasks gsm8k --num_fewshot 5 $CHAT --gen_kwargs max_gen_toks=512
run --tasks minerva_math --num_fewshot 4 $CHAT --gen_kwargs max_gen_toks=512
[ "$MODE" = chat ] && run --tasks ifeval --apply_chat_template --gen_kwargs max_gen_toks=1024
[ "$MODE" = chat ] && run --tasks humaneval_instruct --apply_chat_template --confirm_run_unsafe_code --gen_kwargs max_gen_toks=768
[ "$MODE" = base ] && run --tasks humaneval --confirm_run_unsafe_code
# log-likelihood knowledge/commonsense (no template, same as base card)
run --tasks mmlu --num_fewshot 5
run --tasks arc_challenge,arc_easy,hellaswag,piqa,winogrande,openbookqa,truthfulqa_mc2
echo EVAL_DONE $N
