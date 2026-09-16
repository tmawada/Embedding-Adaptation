#!/usr/bin/env bash
# Query distillation: paired/supervised baselines for Query-DANN v2 (teacher = base BGE-M3 on the formal twin).
# Needs the shared embedding cache (../cache, built by ../prepare.py) and the v2 checkpoint for the reference row.
set -euo pipefail
cd "$(dirname "$0")"

python3 train.py --run_name embed --distill embed                          # label-free, embedding-level
python3 train.py --run_name score --distill score                          # label-free, score-level (listwise)
python3 train.py --run_name embed_infonce --distill embed --w_infonce 1.0  # + relevance labels
python3 train.py --run_name score_infonce --distill score --w_infonce 1.0  # + relevance labels

python3 evaluate.py --reference query_dann_v2 \
  --run query_dann_v2=../runs/v2_dann_g1.0/best \
  --run embed=runs/embed/best --run score=runs/score/best \
  --run embed_infonce=runs/embed_infonce/best --run score_infonce=runs/score_infonce/best \
  --out runs/test_results.json
