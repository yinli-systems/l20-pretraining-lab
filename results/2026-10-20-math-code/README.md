# English math and code: final held-out evaluation

The preselected **SFT step 12,000** checkpoint was evaluated once against
`AliceYin/L20-1B-20B-Base` after training and selection ended. Each model
answered the same **1,791 held-out questions**. The final report completed
October 20, 2026 at **13:02:28 UTC / 17:02:28 Asia/Dubai**.

| Task | Base correct/n | Post-trained correct/n | Change | Paired 95% interval, pp |
|---|---:|---:|---:|---:|
| GSM8K | 10/1223 · 0.82% | 250/1223 · **20.44%** | **+19.62 pp** | [+17.33, +21.91] |
| MATH500 | 6/436 · 1.38% | 76/436 · **17.43%** | **+16.06 pp** | [+12.39, +19.72] |
| HumanEval execution pass rate | 0/132 · 0% | 11/132 · **8.33%** | **+8.33 pp** | [+3.79, +13.64] |

Accuracy improved on all three tasks, but absolute scores remain low,
particularly for code. This does not meet a claim of strong general mathematical
or programming reasoning. No relative multiple is reported for a zero baseline.

These are the remaining held-out portions after removing a development set
of GSM8K 96, MATH500 64 and HumanEval 32 questions. They are **not full official
benchmark scores** and must not be compared directly with differently prompted
or decoded leaderboard results. Final test outputs were not used to choose a
new checkpoint or continue training. General-language retention and safety
were not evaluated in this follow-on study.

## Frozen protocol

- Seed `20261019`, greedy decoding; native context 2048 tokens, left-truncated
  prompt cap 1100; at most `min(1800, 2048 - padded_prompt_length)` new tokens.
- Identical benchmark content and task IDs, with no duplicate IDs or prompts.
  The local audit also checked disjointness from the development IDs/prompts.
  This does not establish absence of every possible semantic contamination.
- The base uses continuation prompts; the post-trained model uses ChatML.
  This is an interface-adapted comparison, not identical rendered prompts or
  an ablation separating formatting gains from reasoning gains.
- Mathematical answer grading uses `math-verify`. Code uses the frozen
  `v2-hashlib-entrypoint` extraction and an empty chroot, unprivileged UID,
  seccomp allowlist, 3-second CPU / 6-second wall / 768-MiB address-space limits.
  Hashlib is preloaded before isolation. No test-guided code-block selection.
- Both evaluations used dynamic KV caching. Scores were recomputed from all
  stored verdicts; this publication did not generate or regrade answers.

Intervals use 10,000 paired bootstrap resamples, seeds 20261019–20261021.
They describe benchmark-question uncertainty, not training-seed variation.
Per-model Wilson intervals are also stored in [metrics.json](metrics.json).
Math answer correctness does not prove the reasoning trace correct; code passes
cover the supplied tests only.

## Selection and failed RFT candidates

| Development checkpoint | GSM8K /96 | MATH500 /64 | HumanEval /32 | Selected |
|---|---:|---:|---:|---|
| SFT 12,000 | 23 | 19 | 2 | **Yes** |
| RFT round 1, 23 updates | 21 | 13 | 2 | No |
| RFT round 2, 23 updates | 17 | 14 | 3 | No |

The second RFT round applied the repaired training quality filter, removing
21/958 candidates and training on 937. It gained one code question over the
selected SFT model but lost math accuracy and did not improve the equal-task
development macro. The final model therefore remained SFT 12,000. Per-task
best scores from different checkpoints are never combined into one model.
See [development evidence](evidence/development-and-rft.json) and the
[training source and audit explanation](../../posttraining/2026-10-20-video-model/README.md).

## Published evidence and verification

```bash
python3 posttraining/2026-10-20-video-model/verify_results.py
```

- [Paired verdicts](paired-verdicts.csv): task IDs, prompt SHA256, and both
  correctness flags. No raw benchmark prompts, answers, tests or completions.
- [Metrics and uncertainty](metrics.json), [base evaluation metadata](evidence/base-final.json),
  [post-trained evaluation metadata](evidence/best-final.json).
- [Original input hashes](evidence/original-input-hashes.json),
  [model file hashes](evidence/final-model-sha256.json), and
  [publication manifest](source-manifest.json).

Public verification recomputes stored scores and intervals and checks hashes;
it cannot independently establish that the original model outputs were graded
correctly without the unpublished raw-output receipts. Those receipts were
locally checked against remote SHA256 values. Model weights, recovery states,
private deployment details and machine-local logs remain outside GitHub.
