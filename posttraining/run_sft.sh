source .venv/bin/activate
export NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GPUS:-0,8,9} OMP_NUM_THREADS=8 EPOCHS=${EPOCHS:-1} MICRO=${MICRO:-8} ACCUM=${ACCUM:-6} LR=${LR:-1e-4}
NP=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
torchrun --nproc_per_node $NP --master_port 29611 train_sft.py
