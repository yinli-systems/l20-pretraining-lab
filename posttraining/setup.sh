set -x
cd /workspace/l20-posttrain
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers==4.56.2" "trl==0.23.0" "datasets>=3" accelerate deepspeed liger-kernel hf_transfer "lm-eval[ifeval,math]==0.4.9" "vllm==0.11.0" wandb sentencepiece
uv pip install flash-attn --no-build-isolation || echo FLASH_FAIL
python -c "import torch,transformers,trl,vllm;print(torch.__version__,transformers.__version__,trl.__version__,vllm.__version__)"
python -c "import flash_attn;print('fa',flash_attn.__version__)"
echo SETUP_DONE
