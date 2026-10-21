# Repository layout and migration

The September 28 refresh separates the language release, multimodal research,
latest results and contributor documentation. It does not rewrite old scientific
records to make the new presentation look cleaner.

| Area | Purpose |
|---|---|
| [pretraining/](../pretraining/README.md) | Original language scripts, reports, artifacts, tests and reproduction bundle |
| [L20-VL-1.2B/](../L20-VL-1.2B/README.md) | Multimodal implementation and historical experiments |
| [results/](../results/README.md) | Dated result packages and study index |
| [docs/](.) | Architecture, limitations and reproducibility |
| [tools/](../tools/) | CPU-only verification tools |
| [tests/](../tests/) | New layout/result regression checks |
| [posttraining/](../posttraining/README.md) | SFT + DPO post-training of the released base (interim, 2026-10-21) |

## Lossless historical relocation

The [migration manifest](layout-migration.json) records each of the 646 files
tracked at commit `d5f3e981ebfcdf84a1612e0e8e5fa175ab2c901b`, its new path and
its SHA-256. The verifier checks every entry. Original training source and
bound receipts remain byte-identical. The former root and VLM landing pages
are preserved as `HISTORY.md`; the previous workflow is archived separately
before adding new CI checks.

Language scripts, tests and evidence moved **together** under `pretraining/`,
so existing sibling paths and the original evidence verifier still work.
Use `cd pretraining` for the historical training commands. Historical external
links may refer to old paths; use this map rather than changing bound records.

## Branch naming

Active work uses `kevin/<topic>`. The existing branches were renamed without
changing their commit objects:

| Previous | Current | Preserved tip |
|---|---|---|
| `codex/vlm-results-20260927` | `kevin/vlm-results-20260927` | `46f897a8cfa98c721d115adf5d858f44483fbc1e` |
| `codex/529m-results` | `kevin/529m-results` | `eb5714d86fb86b2ba608148e9a1fc2298ec6b501` |

These were atomic Git ref moves, not GitHub URL redirects. Existing clones can
fetch with pruning and update their local branch tracking. Closed PR history,
original commit authors and historical source strings are not rewritten.
The unrelated 529M branch was renamed, **not silently merged**.

## Post-merge test stabilization

The original SIGTERM test used a fixed 0.2-second timer; slow initialization could
terminate the entire pytest process before its handler existed. The active test
now sends the real signal from a child process only after the packing iterator
starts and verifies the installed handler. It additionally exercises delayed
tokenizer initialization. The original test is preserved byte-for-byte at
`pretraining/archive/tests/test_pack_data_checkpoint_stop.py`, and its migration
entry points there. Production packing code, numerical results and original
training-source hashes are unchanged.

## Language-results presentation refresh

The repository is now `yinli-systems/l20-pretraining-lab`; the underlying
repository ID and history are unchanged. `assets/` contains one reproducible
comparison figure, not decorative/generated benchmark imagery.
`results/language-model/` exposes the existing 36-checkpoint study, including
stronger competitors, the six-task sensitivity, confidence intervals and
per-task scores. Historical sources remain at their recorded paths and hashes.
