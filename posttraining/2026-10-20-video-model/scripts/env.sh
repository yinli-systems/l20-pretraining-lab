#!/bin/bash
export RUN_ROOT=/workspace/posttrain-l20
export HF_HOME=/dev/shm/l20-posttrain/hf
export HF_DATASETS_CACHE=/dev/shm/l20-posttrain/datasets
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/workspace/posttrain-l20/compile-cache
export TRITON_CACHE_DIR=/workspace/posttrain-l20/triton-cache
export PATH=/venv/main/bin:$PATH
mkdir -p /dev/shm/l20-posttrain /workspace/posttrain-l20/logs
cd /workspace/posttrain-l20
