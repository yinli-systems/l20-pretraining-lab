# Packed SFT with HF Trainer; flash-attn varlen via position_ids (no attention_mask).
import os, sys, numpy as np, torch, faulthandler, signal
faulthandler.register(signal.SIGUSR1, all_threads=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = "/workspace/l20-posttrain"
OUT = os.environ.get("OUT", f"{ROOT}/ckpt/sft")
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "1e-4"))
MICRO = int(os.environ.get("MICRO", "4"))
ACCUM = int(os.environ.get("ACCUM", "6"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))

class Packed(torch.utils.data.Dataset):
    def __init__(self):
        self.ids = np.load(f"{ROOT}/data/sft_ids.npy", mmap_mode="r")
        self.lab = np.load(f"{ROOT}/data/sft_lab.npy", mmap_mode="r")
        self.pos = np.load(f"{ROOT}/data/sft_pos.npy", mmap_mode="r")
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        return {"input_ids": torch.from_numpy(self.ids[i].astype(np.int64)),
                "labels": torch.from_numpy(self.lab[i].astype(np.int64)),
                "position_ids": torch.from_numpy(self.pos[i].astype(np.int64))}

def collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}

def main():
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/init")
    model = AutoModelForCausalLM.from_pretrained(f"{ROOT}/init", torch_dtype=torch.float32,
                                                 attn_implementation="flash_attention_2")
    args = TrainingArguments(
        output_dir=OUT, num_train_epochs=EPOCHS, max_steps=MAX_STEPS,
        per_device_train_batch_size=MICRO, gradient_accumulation_steps=ACCUM,
        learning_rate=LR, lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": 0.1},
        warmup_ratio=0.03, weight_decay=0.01, adam_beta2=0.95, max_grad_norm=1.0,
        bf16=True, tf32=True, use_liger_kernel=True,
        logging_steps=10, save_strategy="steps", save_steps=500, save_total_limit=3,
        save_only_model=True, report_to="none", dataloader_num_workers=4,
        ddp_find_unused_parameters=False, seed=0, remove_unused_columns=False,
    )
    tr = Trainer(model=model, args=args, train_dataset=Packed(), data_collator=collate)
    tr.train()
    tr.save_model(f"{OUT}/final")
    tok.save_pretrained(f"{OUT}/final")

if __name__ == "__main__":
    main()
