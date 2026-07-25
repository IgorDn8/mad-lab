#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/Texas/mad-lab
exec /home/ubuntu/.local/bin/uv run python scripts/run_comparison_suite.py \
  --regime iso-d128-iso1m \
  --iso-batch 1 4 32 128 \
  --iso-max-seq 65536 \
  --iso-lru-impls orig custom_hopscan_autotune \
  --step-ceiling-ms 5000 \
  --per-cell-timeout-s 165 \
  2>&1 | tee run_iso_d128_1m_orig_autotune.log
