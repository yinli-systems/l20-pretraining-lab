# video model: English math and code post-training

This is the source snapshot for a bounded follow-on experiment on
`AliceYin/L20-1B-20B-Base`, revision
`7fc9ab0f50e7faf34c2487fefdf46aa99f4a50b8`.
[Final results and paired verdicts](../../results/2026-10-20-math-code/README.md)
are separate from the earlier pretraining and multimodal experiments.

One NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB reported memory)
was used with a 24-hour experiment deadline, starting October 19, 2026 at
19:48:39 UTC. Backup and shutdown verification continued after that deadline;
this is not a claim of exactly 24 hours of total instance allocation or billing.
Training and selection ended October 20 at 11:58:03 UTC after two RFT rounds
failed to improve the selected development score. Final evaluation finished
at 13:02:28 UTC, about 17.23 elapsed hours after budget start. Elapsed time
includes preparation and evaluation; it is not a measurement of active GPU
training time or energy use. Backup and shutdown completion are not asserted
by this original result snapshot. The [follow-up package](../../results/2026-10-20-qwen-validation/README.md)
records the later Qwen comparison, unlaunched validation pilots and separately
verified backup/shutdown outcome.

## Training

- Full-parameter SFT: BF16 computation, FP32 master weights and AdamW states,
  compiled backbone, fused Liger kernels, micro-batch 22, accumulation 2,
  sequence length 2048, assistant-only loss.
- Three 128-step learning-rate pilots selected `2e-4`. Main SFT completed
  12,635 updates; development selection retained step 12,000.
- 797,876 training conversations, 192,412 packed blocks, 387,229,477
  non-padding tokens and 280,546,530 supervised tokens per prepared epoch.
  Packing efficiency was 98.27%. These are corpus counts, not total exposure
  across all epochs. See the [data manifest](../../results/2026-10-20-math-code/evidence/data-manifest.json).
- Sources include OpenMathInstruct-2, OpenCodeInstruct, NuminaMath-CoT,
  SmolTalk, and short complete OpenR1/OpenCodeReasoning traces. Source
  revisions and filtering are recorded in the preparation scripts. Released
  R1 code traces were syntax checked, not all independently executed.
- Conversations are causally concatenated with EOS separators; packed
  conversations share the causal context. Long answers were filtered as
  whole examples. Prompt-grouped validation was kept separate.
- RFT means rejection-sampling fine-tuning, not PPO/GRPO. Four candidates
  per training question were checked by mathematical answer verification
  or isolated execution. Accepted completions, references and original-data
  replay were used for two short rounds, each with 23 optimizer updates.
  Neither candidate exceeded the selected SFT model.

A configuration benchmark measured about 27,996 packed tokens/s over seven
timed steps. This synthetic benchmark is not a full-run average, supervised
token throughput, or proof of globally optimal utilization.

## Corrections and negative outcomes

The first code evaluation omitted a `hashlib` preload before entering its
empty chroot. Revision `v2-hashlib-entrypoint` fixed that false negative and
uses the first complete code block defining the requested function, preserving
the supplied prompt context. Selection is deterministic and never chooses a
block by test success. Official development references passed 32/32 after the
fix. The same stored first-round outputs changed from 2/32 to 3/32; that was
a grading correction, not a training gain. All reported comparisons here use v2.

An RFT regression audit regraded 384 existing development outputs without
disagreement and reconstructed all 542 first-round packed blocks exactly.
It also found unfinished accepted answers and three independently confirmed
bad training questions/labels. `rft-quality-v2` added structural completeness,
explicit final-answer and known-bad-question checks at the next training
handoff. Round two retained 937/958 candidates (440 math, 497 code), excluding
21. It still did not improve overall development performance. The rounds used
different sampled data; their difference is not a controlled causal estimate
of this fix. Answer matching does not certify every reasoning step.

## Inspect and verify

```bash
python3 posttraining/2026-10-20-video-model/verify_results.py
```

This CPU-only command checks published file hashes and recomputes scores,
paired gains/losses and confidence intervals from the 1,791 paired verdicts.
It performs no model inference or code execution. The original raw-output
comparison utility is also included in `scripts/compare_final_evaluation.py`;
it requires the private local receipt layout, which is not included here.

`source-provenance.json` binds the source files, and `environment.freeze.txt`
records installed package versions from the run. One machine-local packaging
wheel reference was omitted; this freeze is an environment inventory, not a
portable dependency lock or a claim of one-command GPU reproduction.

The historical driver contains the original fixed deadline and run-specific
Linux paths. It is an execution snapshot: do not launch it unchanged. Local
SSH/launchd monitor deployment is excluded. Model weights, optimizer states,
raw datasets, prompts/completions and machine-local logs are not published.
No post-trained weight release or general-capability improvement is claimed.
