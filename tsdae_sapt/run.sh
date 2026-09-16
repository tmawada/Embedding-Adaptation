#!/usr/bin/env bash
# SAPT: TSDAE full fine-tuning of BGE-M3 on informal Indonesian reviews, then zero-shot retrieval evaluation.
# No query adapter is used here.
set -euo pipefail
cd "$(dirname "$0")/.."   # bge_dann/

python3 tsdae_sapt/train_sapt.py                              # ~5 min, saves tsdae_sapt/output/bge-m3-sapt-informal-id (~2.3 GB)
python3 tsdae_sapt/prepare_sapt_cache.py --mode query_only    # seconds: SAPT queries vs original passage index
python3 tsdae_sapt/prepare_sapt_cache.py --mode symmetric     # ~15 min, +1 GB: SAPT for queries and passages
python3 tsdae_sapt/evaluate_sapt.py --split test
python3 tsdae_sapt/evaluate_sapt.py --split dev
