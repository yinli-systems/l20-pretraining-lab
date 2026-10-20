# Qwen comparison and unlaunched validation pilots

This package adds measured comparator results and an explicit preparation failure outcome. It contains no new Chat or 48-hour pilot model.

## Same-question comparator results

| Model | GSM8K portion | MATH500 portion | HumanEval portion |
|---|---:|---:|---:|
| L20 base | 10/1223 (0.82%) | 6/436 (1.38%) | 0/132 (0.00%) |
| L20 SFT12000 Instruct | 250/1223 (20.44%) | 76/436 (17.43%) | 11/132 (8.33%) |
| Qwen2.5-0.5B-Instruct | 608/1223 (49.71%) | 135/436 (30.96%) | 48/132 (36.36%) |
| Qwen2.5-1.5B-Instruct | 899/1223 (73.51%) | 230/436 (52.75%) | 75/132 (56.82%) |

| Qwen minus L20 SFT12000 | GSM8K | MATH500 | HumanEval |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | +29.27 pp | +13.53 pp | +28.03 pp |
| Qwen2.5-1.5B-Instruct | +53.07 pp | +35.32 pp | +48.48 pp |

Each model answered the same 1,791 project-held-out questions (1,223 / 436 / 132). These are held-out portions, not full official benchmark scores. L20 remains substantially behind both Qwen models on every task. No checkpoint or decoding setting was selected after observing this comparison.

The frozen v2-hashlib-entrypoint generator/grader used greedy BF16 decoding, seed 20261019, context 2048, prompt cap 1100 and at most 1800 new tokens constrained by remaining context. Math uses math-verify; code is actually run in the original isolated executor. L20 base uses continuation prompts; L20 SFT and Qwen use their respective chat interfaces. Qwen native templates may add a default system message; tokenization, actual truncation and numerical EOS IDs differ. Rendered prompt strings are not identical.

Pinned Qwen revisions, all four models' Wilson intervals, paired gains/losses and fixed-seed 10,000-draw bootstrap intervals are in [metrics.json](metrics.json). [Paired verdicts](paired-verdicts.csv) permit CPU recounting without publishing questions or completions. This publication verifier does not independently regenerate or regrade the raw outputs.

## Validation preparation ended without training

The follow-on 48-hour three-pilot clock never started. No GPU preflight, new training, quality-approved production native corpus or new pilot weights were produced. At the original preparation bound (2026-10-20 19:48:39 UTC), the existing process stopped rather than relax the data, leakage or storage gates.

- Exhaustive review of 245 computational math candidates retained 186 unedited traces; old-data screening left 98 candidates / 62,010 supervised tokens. New-test screening and final quality approval were still missing. This cannot support the planned 100k-question / hundreds-of-millions-of-tokens design.
- The unchanged 30,000-row general SFT candidate pool was blocked after five unambiguous arithmetic/algebra failures in the reviewed part of a fixed 200-conversation sample (35 read). 5/200 is a conservative sample lower bound, not a source-wide error estimate. Earlier failed math-derived candidate pools were likewise blocked.
- Code collection ended with 90,832 quarantined candidates. The last completed custom code-suite proposal had only 172 development / 334 confirmation items, below the required 400 / 800; the bounded suite service exited 124 before freeze. Counts of candidates/proposals are not model capability scores.
- Raw CPT buffers and old-overlap token-accounting checks completed, but their usable-token upper bounds are not approved packed training data. Source review, new-test exclusion, complete native packing/replay and full training-backup capacity approval remained unfinished.
- CPU objective/cursor and 52-sample / 17-block packing controls passed. Actual GPU fused loss/gradient/Adam throughput preflight did not run. These controls do not establish production readiness.
- Historical weights/reference/recovery remained backed up. New preparation terminal files and 18 lossless raw/index archives were verified locally before the stop request (19:58:59 UTC); the provider console subsequently displayed Inactive. Backup/verification continued after the 19:48:39 experiment deadline, so this is not a claim that total instance allocation or billing was limited to exactly 24 hours. Storage charges can remain. Machine-local paths, credentials and logs are omitted.

The user deferred the proposed Chat adaptation. The existing SFT12000 Instruct weights support ChatML formatting, but no separately trained or validated general Chat release is claimed.

See [preparation-outcome.json](preparation-outcome.json) and the [source snapshot](../../posttraining/2026-10-20-followup/README.md).

## Verify

```bash
python3 posttraining/2026-10-20-followup/verify_results.py
```
