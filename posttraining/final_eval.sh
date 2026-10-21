# Evaluate SFT final and DPO final on any free GPU as soon as each is ready; retry until EVAL_DONE.
cd /workspace/l20-posttrain; source .venv/bin/activate
pick() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<500{print $1}' | while read g; do [ -e /tmp/evgpu_$g ] || { echo $g; break; }; done; }
run_one() { # name model donefile
  until [ -e "$3" ]; do sleep 5; done
  until grep -q EVAL_DONE logs/final_$1.log 2>/dev/null; do
    g=""; while [ -z "$g" ]; do g=$(pick); [ -z "$g" ] && sleep 1; done
    touch /tmp/evgpu_$g; echo "$(date) eval $1 on GPU $g"
    CUDA_VISIBLE_DEVICES=$g timeout 3h python eval_all.py $2 eval/$1 chat > logs/final_$1.log 2>&1
    rm -f /tmp/evgpu_$g; sleep 5
  done; echo "$(date) $1 EVAL_DONE"; }
run_one sft_final $PWD/ckpt/sft/final ckpt/sft/DONE &
run_one dpo_final $PWD/ckpt/dpo/final ckpt/dpo/DONE &
wait
