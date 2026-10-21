# Started early. DPO ranks wait (inside python) for ckpt/sft/DONE, then claim GPUs instantly.
cd /workspace/l20-posttrain; source .venv/bin/activate
( until [ -e ckpt/sft/final/model.safetensors ] && [ -e ckpt/sft/final/tokenizer_config.json ] && ! pgrep -f train_sft.py >/dev/null; do sleep 1; done
  touch ckpt/sft/DONE; echo "$(date) SFT DONE" ) &
export NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,4,5,6 OMP_NUM_THREADS=8 WAIT_FOR=$PWD/ckpt/sft/DONE SFT=$PWD/ckpt/sft/final OUT=$PWD/ckpt/dpo
torchrun --nproc_per_node 4 --master_port 29621 train_dpo.py && touch ckpt/dpo/DONE
echo "$(date) DPO exit $?"
