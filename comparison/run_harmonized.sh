#!/usr/bin/env bash
# Apple-to-apple comparison: identical training protocol for every trained model.
#
# Shared protocol (all trained models): 1350 optimiser steps | effective batch 64 | AdamW lr 3e-5 -> 1e-6
# cosine (10% warmup) | weight decay 0.01 | grad clip 1.0 | bf16 | max_length 128 | seed 42 |
# checkpoint = best dev nDCG@10 on informal queries, evaluated every 50 steps.
# Method-specific knobs keep their own values (gamma/mu/tau/bottleneck for adapters, del_ratio for SAPT,
# temperature/shots for the LLM); BGE-M3 base and LLM normalisation train nothing.
set -euo pipefail
cd "$(dirname "$0")/.."   # bge_dann/

H=comparison/runs/harmonized
SHARED="--lr 3e-5 --lr_min 1e-6 --weight_decay 0.01 --batch_size 64 --eval_every 50 --seed 42 --mixed_precision bf16"

# 1. SAPT towers under the shared protocol (full fine-tuning; ~1.5 h each).
python3 -u comparison/train_sapt_harmonized.py --source reviews --output_dir $H/sapt_reviews/model
python3 -u comparison/train_sapt_harmonized.py --source queries --output_dir $H/sapt_queries/model

# 2. Query caches for those towers; passages remain the original frozen index (symlinked).
python3 -u tsdae_sapt_queries/prepare_tsdae_cache.py --model $H/sapt_reviews/model --out_cache $H/sapt_reviews/cache
python3 -u tsdae_sapt_queries/prepare_tsdae_cache.py --model $H/sapt_queries/model --out_cache $H/sapt_queries/cache

# 3. Adapters under the shared protocol, on the base tower and on each SAPT tower.
for pair in "base:cache" "sapt_reviews:$H/sapt_reviews/cache" "sapt_queries:$H/sapt_queries/cache"; do
  name=${pair%%:*}; C=${pair#*:}
  for g in 0.1 0.5 1.0; do
    python3 -u query_dann/train.py --cache_dir "$C" --output_dir $H/$name/query_dann_v2 --run_name g$g --gamma $g --epochs 30 $SHARED
  done
  for d in score embed; do
    python3 -u query_distillation/train.py --cache_dir "$C" --output_dir $H/$name/query_distillation \
      --run_name $d --distill $d --total_steps 1350 $SHARED
  done
done

# 4. One table (gamma picked on dev, as for every other method).
BEST=$(python3 -c "import json,glob;ps=glob.glob('$H/base/query_dann_v2/g*/best/config.json');print(max(ps,key=lambda p:json.load(open(p))['dev']['nDCG@10']).rsplit('/',1)[0])")
python3 -u comparison/compare_all.py --out_dir $H \
  --dann "$BEST" \
  --distill_score $H/base/query_distillation/score/best \
  --distill_embed $H/base/query_distillation/embed/best \
  --sapt_reviews_cache $H/sapt_reviews/cache --sapt_reviews_dann $H/sapt_reviews/query_dann_v2 \
  --sapt_reviews_distill $H/sapt_reviews/query_distillation/score/best \
  --sapt_queries_cache $H/sapt_queries/cache --sapt_queries_dann $H/sapt_queries/query_dann_v2 \
  --sapt_queries_distill $H/sapt_queries/query_distillation/score/best \
  --sapt_reviews_symmetric /nonexistent
