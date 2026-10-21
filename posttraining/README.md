# Post-training L20-1B-20B-Base

**Status: interim (2026-10-21).** SFT is complete; DPO and the full final-model
evaluation are still running. Numbers below are what has been measured so far.

This directory turns the released base model
[`AliceYin/L20-1B-20B-Base`](https://huggingface.co/AliceYin/L20-1B-20B-Base)
into a chat model with supervised fine-tuning (SFT) followed by DPO. It is a
separate study from the frozen pretraining comparison; nothing under
`pretraining/` or `results/` was changed.

> **Not a protocol result.** This is an exploratory run outside the
> preregistered [post-training protocol](../pretraining/posttraining_protocol.json)
> (2026-09-12-v1). It skips that protocol's stage-0 parent-selection gate,
> its own decontamination and provenance audit (it relies on the public
> datasets' decontamination), base-causal replay, and the pilot learning-rate
> sweep. Do not cite it as satisfying any protocol stage or promotion gate.

## Recipe

| Stage | Data | Setup |
|---|---|---|
| Chat format | — | ChatML; `<\|im_start\|>`/`<\|im_end\|>` added as special tokens (embeddings mean-initialised, vocab padded to 32,064) |
| SFT | 2.05M conversations, 1.32B tokens (see below) | 1 epoch, packed 2,048-token rows, loss on assistant tokens only, LR 1e-4 cosine → 1e-5, global batch 144 × 2,048, 4× A40, ~10 h |
| DPO | `HuggingFaceH4/ultrafeedback_binarized` (61k pairs) | 1 epoch, β = 0.1, LR 1e-6, global batch 64 |

SFT mixture (after dropping conversations longer than 2,048 tokens):

| Source | Conversations | Tokens | Assistant tokens |
|---|---:|---:|---:|
| `HuggingFaceTB/smoltalk` (all) | 988,660 | 759M | 558M |
| `nvidia/OpenMathInstruct-2` (600k random from `train_1M`) | 599,591 | 270M | 223M |
| `OpenCoder-LLM/opc-sft-stage2` (all subsets) | 434,392 | 280M | 176M |
| `allenai/tulu-3-sft-personas-instruction-following` | 29,710 | 12M | 10M |

## Results so far

Scores are percentages. Comparison models were evaluated with the same scripts
on the same machine.

| Model | MATH-500 (0-shot) | GSM8K (0-shot) | GSM8K (5-shot) | HumanEval | IFEval (prompt, loose) | MMLU (5-shot) |
|---|---:|---:|---:|---:|---:|---:|
| L20-1B-20B-Base (before post-training) | — | — | 2.2 | 0.6 | — | 25.5 |
| SFT step 1000 (22%) | 2.6 | 3.0 | 3.9 | | | |
| SFT step 2000 (45%) | 4.6 | 5.2 | | | | |
| **SFT final** | **5.0** | **8.0** | pending | pending | pending | pending |
| DPO final | pending | pending | pending | pending | pending | pending |
| TinyLlama-1.1B-Chat-v1.0 | 2.2 | 2.1 | 2.7 | 8.5 | 16.8 | 25.0 |
| Qwen2.5-0.5B-Instruct | 33.2 | 49.4 | 40.5 | 32.3 | 29.2 | 46.9 |

- **0-shot math** (`math_eval.py`): chat template, greedy decoding, the model is
  asked for a `\boxed{}` answer, graded with `math-verify`. This harness
  reproduces Qwen2.5-0.5B-Instruct's published MATH-500 score (33.2 vs ~34).
- **5-shot GSM8K, MMLU, HumanEval, IFEval**: `lm-eval` 0.4.13 with the vLLM
  backend; chat models use their chat template (multi-turn few-shot). The base
  model is evaluated without a template, and its HumanEval is the completion
  task rather than `humaneval_instruct`.
- `lm-eval`'s `minerva_math` exact-match is **not reported**: it fails to parse
  chat-model answers (it gives Qwen2.5-0.5B-Instruct 1.6%). Raw values are kept
  in [results/results-2026-10-21.json](results/results-2026-10-21.json).

**Reading the numbers.** After SFT, the model beats TinyLlama-1.1B-Chat (same
architecture and size, 3T pretraining tokens) on zero-shot math by about 2×
(MATH-500) and 4× (GSM8K). It is far behind Qwen2.5-0.5B-Instruct, which was
pretrained on roughly 900× more tokens. Post-training mostly teaches format
and procedure. With 20B pretraining tokens, arithmetic and knowledge remain the
limit, and MMLU is not expected to move much from chance.

## Running it

The scripts assume the working directory `/workspace/l20-posttrain` and
`HF_HOME` set. They were run on a shared 10× A40 host.

```bash
bash setup.sh                    # venv + pinned packages (requirements.txt)
python dl.py                     # base model + datasets
python prep_sft.py               # chat tokens/template, tokenise, pack -> data/*.npy
GPUS=0,4,5,6 MICRO=4 ACCUM=9 bash run_sft.sh
bash run_dpo.sh                  # waits for SFT, then claims the same GPUs
bash final_eval.sh               # evaluates SFT and DPO finals on any free GPU
```

`eval_baselines.sh` produced the base, TinyLlama and Qwen baseline rows with
the `lm-eval` CLI. `eval_all.py` produces the same benchmarks inside one
process, and `ckpt_eval.sh` evaluates an intermediate checkpoint.

## Operational notes

- **NCCL P2P must be disabled** on this host (`NCCL_P2P_DISABLE=1`). With P2P
  on, DDP hangs in the first collective for GPUs behind the same PCIe switch.
- The host is shared, and other jobs claim any GPU the moment it is idle.
  `train_dpo.py` therefore starts early and reserves GPU memory within ~1 s of
  SFT exiting. `eval_all.py` keeps one vLLM engine for every benchmark, so the
  GPU is not released between tasks.
- `liger-kernel` must stay at 0.6.x for `transformers` 4.56. `numpy` must stay
  below 2.3 for vLLM 0.11's numba dependency.
