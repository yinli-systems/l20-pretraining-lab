# Comparator and validation-preparation source snapshot

The [dated results package](../../results/2026-10-20-qwen-validation/README.md)
contains the completed Qwen comparison and the unlaunched three-pilot outcome.
The existing [Instruct training source](../2026-10-20-video-model/README.md)
and its frozen evaluator remain unchanged.

## Included source

- `qwen/`: sequential pinned Qwen comparison, native tokenizer/template/EOS
  adaptation, hash-verifying collector, and paired comparison utilities.
- `validation48/`: bounded candidate/CPT acquisition, old-data overlap indices,
  source-repair attempts, conservative curation, native objectives, packing,
  CPU checks, staged training/driver guards, and backup/monitor helpers.
- `deployment/stop_instance.py`: updated verified-data backup guard.

This is a historical implementation snapshot, not a production-readiness or
one-command reproduction claim. The three pilots never started. The published
CPU controls passed; real-model GPU fused-gradient/throughput preflight,
production data quality, fresh-suite freeze and training storage gates did not.
All scripts retain historical expired deadlines and Linux staging paths.
Do not launch them unchanged or treat candidate acquisition as trained weights.
Raw corpora, review text, questions, completions, weights, optimizer states and
private deployment receipts are not included.

`source-provenance.json` binds both the original private source bytes and the
published bytes. Only the SSH collector/monitor configuration was adapted for
publication: `VIDEO_MODEL_SSH_TARGET` must be supplied explicitly, with optional
`VIDEO_MODEL_SSH_PORT` (default 22). The installer propagates these settings to
launchd. No private host, port, SSH key or credential value is published. These
adaptations were not deployed back onto the completed experiment.

Most validation preparation modules require the original Linux layout, source
revisions, review receipts and tokenizer, plus Python 3.11+ (file_digest), NumPy,
Transformers, datasets, math-verify and the original sandbox environment.
The pure publication verifier needs only Python 3.9+ and the standard library:

```bash
python3 posttraining/2026-10-20-followup/verify_results.py
```

It checks publication/source hashes and recomputes the four-model counts,
Wilson intervals and paired Qwen-minus-L20 intervals from 1,791 verdict rows.
It performs no inference, code execution or independent raw-answer grading.
Chat adaptation was deferred by the user and has not been trained.
