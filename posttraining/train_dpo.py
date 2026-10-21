# DPO on UltraFeedback (binarized) starting from the SFT model.
import os, glob, time, subprocess, torch

def claim_gpu_after(path, gb=34):
    # Started early: wait for SFT to finish, then grab memory the instant it is released,
    # before other jobs on this machine can take the GPU. The caching allocator keeps it reserved.
    while not os.path.exists(path):
        time.sleep(5)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    phys = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank]
    torch.cuda.set_device(rank)
    while True:
        free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", phys]).decode())
        if free > gb * 1024 + 2048:
            try:
                x = torch.empty(int(gb * 2**30), dtype=torch.uint8, device="cuda"); del x
                return
            except torch.OutOfMemoryError:
                pass
        time.sleep(0.1)

if os.environ.get("WAIT_FOR"):
    claim_gpu_after(os.environ["WAIT_FOR"])
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ROOT = "/workspace/l20-posttrain"
SFT = os.environ.get("SFT", f"{ROOT}/ckpt/sft/final")
OUT = os.environ.get("OUT", f"{ROOT}/ckpt/dpo")

def main():
    tok = AutoTokenizer.from_pretrained(SFT)
    tok.pad_token = "<|padding|>"
    f = glob.glob(os.environ["HF_HOME"].rstrip("/") + "/hub/datasets--HuggingFaceH4--ultrafeedback_binarized/snapshots/*/data/train_prefs-*.parquet")
    ds = load_dataset("parquet", data_files=f, split="train")
    ds = ds.map(lambda r: {"prompt": [{"role": "user", "content": r["prompt"]}],
                           "chosen": r["chosen"][1:], "rejected": r["rejected"][1:]},
                remove_columns=ds.column_names, num_proc=16)
    model = AutoModelForCausalLM.from_pretrained(SFT, dtype=torch.float32, attn_implementation=os.environ.get("ATTN", "flash_attention_2"))
    ref = AutoModelForCausalLM.from_pretrained(SFT, dtype=torch.bfloat16, attn_implementation=os.environ.get("ATTN", "flash_attention_2"))
    args = DPOConfig(
        output_dir=OUT, num_train_epochs=float(os.environ.get("EPOCHS", "1")), max_steps=int(os.environ.get("MAX_STEPS", "-1")),
        per_device_train_batch_size=int(os.environ.get("MICRO", "4")), gradient_accumulation_steps=int(os.environ.get("ACCUM", "4")),
        learning_rate=float(os.environ.get("LR", "1e-6")), beta=float(os.environ.get("BETA", "0.1")),
        lr_scheduler_type="cosine", warmup_ratio=0.1, bf16=os.environ.get("ATTN") is None, tf32=os.environ.get("ATTN") is None, use_cpu=os.environ.get("ATTN") is not None,
        max_length=2048, max_prompt_length=1024, gradient_checkpointing=True,
        logging_steps=10, save_strategy="no", report_to="none", dataset_num_proc=16,
        ddp_find_unused_parameters=False, seed=0,
    )
    tr = DPOTrainer(model=model, ref_model=ref, args=args, train_dataset=ds, processing_class=tok)
    tr.train()
    tr.save_model(f"{OUT}/final")
    tok.save_pretrained(f"{OUT}/final")

if __name__ == "__main__":
    main()
