#!/usr/bin/env bash
# Embedding model comparison: BGE-M3 vs multilingual-e5-large-instruct.
# Identical data splits, methods, hyperparameters and step budgets as the BGE-M3 experiments.
set -euo pipefail
cd "$(dirname "$0")/.."   # bge_dann/ (the training scripts resolve data/ relative to it)

E=encoder_comparison
ENC=me5_large_instruct
C=$E/cache/$ENC
R=$E/runs/$ENC

# 1. Embedding cache for mE5-large-instruct (~15 min; corpus encoding is skipped if already done).
python3 $E/prepare_encoder.py --encoder $ENC

# 2. Zero-shot robustness of both encoders.
python3 $E/zero_shot.py

# 3. The three methods on mE5-large-instruct: existing scripts, unchanged, pointed at the new cache.
for g in 0 0.1 0.5 1.0; do
  python3 query_dann/train.py --cache_dir $C --output_dir $R/query_dann_v2 --run_name g$g --gamma $g
done
python3 query_distillation/train.py --cache_dir $C --output_dir $R/query_distillation --run_name embed --distill embed
python3 query_distillation/train.py --cache_dir $C --output_dir $R/query_distillation --run_name score --distill score
python3 query_distillation/train.py --cache_dir $C --output_dir $R/query_distillation --run_name embed_infonce --distill embed --w_infonce 1.0
python3 query_distillation/train.py --cache_dir $C --output_dir $R/query_distillation --run_name score_infonce --distill score --w_infonce 1.0
for g in 0 0.1 0.5 1.0; do
  python3 uda_query_dann/train.py --cache_dir $C --output_dir $R/uda_query_dann --run_name disjoint_g$g --gamma $g --target_split disjoint
done
for g in 0 1.0; do
  python3 uda_query_dann/train.py --cache_dir $C --output_dir $R/uda_query_dann --run_name overlap_g$g --gamma $g --target_split overlap
done

# 4. Test evaluation with bootstrap tests, both encoders through the same script.
python3 $E/evaluate_methods.py --encoder bge_m3
python3 $E/evaluate_methods.py --encoder $ENC

# 5. One comparison table -> encoder_comparison/runs/summary.md
python3 $E/summary.py
