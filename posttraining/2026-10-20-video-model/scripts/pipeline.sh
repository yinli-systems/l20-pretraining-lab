#!/bin/bash
set -euo pipefail
source /workspace/posttrain-l20/env.sh
exec python pipeline.py
