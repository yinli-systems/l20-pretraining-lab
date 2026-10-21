# Post-training L20-1B-20B-Base

**Status (2026-10-21):** SFT complete and fully evaluated. DPO was started
but stopped at 28% (step 268 of 956) to free the GPUs for evaluation; no DPO
model is reported. **The result of this run is the SFT model.**

This directory turns the released base model
[`AliceYin/L20-1B-20B-Base`](https://huggingface.co/AliceYin/L20-1B-20B-Base)
into a chat model with supervised fine-tuning (SFT). It is a
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
| DPO (stopped, not reported) | `HuggingFaceH4/ultrafeedback_binarized` (61k pairs) | planned 1 epoch, β = 0.1, LR 1e-6, global batch 64; stopped at step 268/956 |

SFT mixture (after dropping conversations longer than 2,048 tokens):

| Source | Conversations | Tokens | Assistant tokens |
|---|---:|---:|---:|
| `HuggingFaceTB/smoltalk` (all) | 988,660 | 759M | 558M |
| `nvidia/OpenMathInstruct-2` (600k random from `train_1M`) | 599,591 | 270M | 223M |
| `OpenCoder-LLM/opc-sft-stage2` (all subsets) | 434,392 | 280M | 176M |
| `allenai/tulu-3-sft-personas-instruction-following` | 29,710 | 12M | 10M |

## Results

Scores are percentages. All rows were measured with the same scripts on the
same machine.

**Chat / reasoning**

| Model | MATH-500 (0-shot) | GSM8K (0-shot) | GSM8K (5-shot) | HumanEval (chat) | IFEval (prompt, loose) | MMLU (5-shot) |
|---|---:|---:|---:|---:|---:|---:|
| L20-1B-20B-Base (before post-training) | — | — | 2.2 | 0.6* | — | 25.5 |
| SFT step 1000 (22%) | 2.6 | 3.0 | 3.9 | | | |
| SFT step 2000 (45%) | 4.6 | 5.2 | | | | |
| **SFT final** | **4.8** | **8.0** | **10.4** | **15.9** | **35.1** | **26.4** |
| TinyLlama-1.1B-Chat-v1.0 | 2.2 | 2.1 | 2.7 | 4.9 | 16.8 | 25.0 |
| Qwen2.5-0.5B-Instruct | 33.2 | 49.4 | 40.5 | 32.9 | 29.2 | 46.9 |

\*Base model: completion-style `humaneval`, no chat template.

**Zero-shot knowledge / commonsense** (log-likelihood, no chat template)

| Model | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | OpenBookQA | TruthfulQA mc2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L20-1B-20B-Base | 33.9 | 64.1 | 45.2 | 69.5 | 52.2 | 34.2 | 36.9 |
| SFT final | 30.6 | 55.3 | 44.0 | 68.2 | 53.2 | 35.4 | 36.7 |
| TinyLlama-1.1B-Chat-v1.0 | 32.9 | 54.8 | 60.5 | 74.4 | 60.5 | 35.6 | 37.4 |
| Qwen2.5-0.5B-Instruct | 33.6 | 59.1 | 52.4 | 70.2 | 56.0 | 34.8 | 41.8 |

How each column was measured:

- **0-shot math** (`math_eval.py`): chat template, greedy decoding, the model is
  asked for a `\boxed{}` answer, graded with `math-verify`. This harness
  reproduces Qwen2.5-0.5B-Instruct's published MATH-500 score (33.2 vs ~34).
  vLLM batching makes greedy decoding slightly non-deterministic: a separate
  run of the SFT final gave 5.0 on MATH-500.
- **HumanEval (chat)** (`code_eval.py`): chat prompt with no assistant
  prefill, greedy decoding; the first code block is run against the official
  tests. `lm-eval`'s `humaneval_instruct` prefills the function signature and
  expects the model to continue it. The SFT model replies with complete code
  blocks, so under that prefill it writes tests instead and scores 0.0. The
  chat harness gives Qwen2.5-0.5B-Instruct 32.9, close to its 32.3 on
  `humaneval_instruct`.
- **5-shot GSM8K, MMLU, IFEval, zero-shot tasks**: `lm-eval` 0.4.13 with the
  vLLM backend. Chat models use their chat template for generative tasks
  (multi-turn few-shot). The SFT-final MMLU is the sample-weighted mean of the
  57 subjects, run as 8 parallel jobs.
- `lm-eval`'s `minerva_math` exact-match is **not reported**: it fails to parse
  chat-model answers (it gives Qwen2.5-0.5B-Instruct 1.6%). All raw values are
  in [results/results-2026-10-21.json](results/results-2026-10-21.json).

**Reading the numbers.**

- **Against TinyLlama-1.1B-Chat** (same architecture and size, 3T pretraining
  tokens), the SFT model is ahead on every chat/reasoning column: about 4×
  on GSM8K, 3× on HumanEval, and 2× on IFEval and MATH-500.
- **Against Qwen2.5-0.5B-Instruct**, it is ahead on IFEval (35.1 vs 29.2) and
  far behind on math, code and MMLU. Qwen was pretrained on roughly 900× more
  tokens.
- **SFT costs some zero-shot commonsense**, mainly ARC-E (−8.8) and ARC-C
  (−3.3). TinyLlama-Chat stays ahead on HellaSwag, PIQA and WinoGrande; that
  gap comes from pretraining.
- **The limit is pretraining scale.** Post-training mostly teaches format and
  procedure; arithmetic and knowledge still come from the 20B-token base.

## Running it

The scripts assume the working directory `/workspace/l20-posttrain` and
`HF_HOME` set. They were run on a shared 10× A40 host.

```bash
bash setup.sh                    # venv + pinned packages (requirements.txt)
python dl.py                     # base model + datasets
python prep_sft.py               # chat tokens/template, tokenise, pack -> data/*.npy
GPUS=0,4,5,6 MICRO=4 ACCUM=9 bash run_sft.sh
bash run_dpo.sh                  # waits for SFT, then claims the same GPUs (stopped early in this run)
PART=math python eval_all.py ckpt/sft/final eval/sft_final chat   # also gen | mmlu | cs, one GPU each
python code_eval.py ckpt/sft/final eval/sft_final/codechat.json
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
