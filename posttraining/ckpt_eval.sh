# Eval one checkpoint on the first free non-training GPU (retry until done).
cd /workspace/l20-posttrain; source .venv/bin/activate; C=$1; N=$2
until grep -q EVAL_DONE logs/eval_$N.log 2>/dev/null; do
  g=""; while [ -z "$g" ]; do g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=0&&$1!=4&&$1!=5&&$1!=6&&$2<500{print $1; exit}'); [ -z "$g" ] && sleep 1; done
  echo "$(date) eval $N on GPU $g"
  CUDA_VISIBLE_DEVICES=$g timeout 3h python eval_all.py $PWD/$C eval/$N chat > logs/eval_$N.log 2>&1
  sleep 5
done; echo "$(date) $N EVAL_DONE"
