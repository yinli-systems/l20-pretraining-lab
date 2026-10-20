#!/bin/bash
set -euo pipefail
source /workspace/posttrain-l20/env.sh
python prepare.py benchmarks > logs/benchmarks.log 2>&1
jobs=()
for source in math code numina general; do
    python prepare.py "$source" > "logs/prepare-$source.log" 2>&1 &
    jobs+=("$!")
done
failed=0
for job in "${jobs[@]}"; do wait "$job" || failed=1; done
test "$failed" = 0
python prepare.py pack > logs/pack.log 2>&1
