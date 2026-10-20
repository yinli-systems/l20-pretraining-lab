"""CPU-only native shifted-CE, gradient accumulation and cursor parity checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from objective import make_labels, optimizer_step_budget, ResumableBatches


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261020)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=2048, attention_dropout=0.)
    model = LlamaForCausalLM(cfg).float().eval()
    ids = torch.randint(1, 63, (3, 24))
    ids[:, 20:] = 0
    masks = {'cpt': torch.arange(24)[None, :].expand(3, -1) < 20,
             'sft': (torch.arange(24)[None, :].expand(3, -1) >= 8) &
                    (torch.arange(24)[None, :].expand(3, -1) < 20)}
    # A genuine token zero must remain a trainable target, while padded zeros
    # remain ignored: objective masks, rather than token IDs, decide the labels.
    ids[0, 10] = 0
    results = {}
    for objective, mask in masks.items():
        labels = make_labels(ids, mask, objective)
        assert labels[0, 10] == 0 and (labels[:, 20:] == -100).all()
        full = model(input_ids=ids, labels=labels, use_cache=False)
        manual = torch.nn.functional.cross_entropy(full.logits[:, :-1].reshape(-1, 64),
                                                   labels[:, 1:].reshape(-1))
        torch.testing.assert_close(full.loss, manual, atol=1e-6, rtol=1e-6)
        model.zero_grad(set_to_none=True)
        full.loss.backward()
        reference = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        total = int((labels[:, 1:] != -100).sum())
        for i in range(3):
            loss = model(input_ids=ids[i:i+1], labels=labels[i:i+1], use_cache=False).loss
            count = int((labels[i:i+1, 1:] != -100).sum())
            (loss * count / total).backward()
        for n, p in model.named_parameters():
            torch.testing.assert_close(p.grad, reference[n], atol=2e-6, rtol=2e-4)
        results[objective] = dict(native_loss=float(full.loss.detach()),
                                  shifted_supervised_tokens=total, loss_and_gradient_parity=True)
    sampler = ResumableBatches(101, 7, 20261020)
    all_batches = list(iter(sampler))
    sampler.consumed = 6
    assert list(iter(sampler)) == all_batches[6:]
    sampler.consumed = sampler.per_epoch
    assert list(iter(sampler)) != all_batches
    assert list(iter(sampler)) == list(iter(ResumableBatches(101, 7, 20261020, sampler.per_epoch)))
    assert optimizer_step_budget(100, 22, 2, 350_000_000, 2) == 4
    assert optimizer_step_budget(100, 22, 2, 800_000_000) == 800_000_000 // 90112
    print(json.dumps(dict(passed=True, cpu_only=True, native_transformers=True,
                         tests=results, deterministic_cursor=True,
                         budgets_floor_without_epoch_overrun=True,
                         gpu_fused_kernel_parity='pending before real training'), indent=2))


if __name__ == '__main__':
    main()
