#!/usr/bin/env bash
# Full pipeline: cache frozen BGE-M3 embeddings -> train -> evaluate on held-out test queries.
set -euo pipefail
cd "$(dirname "$0")/.."   # bge_dann/ (shared prepare.py / eval.py live there)

# 1. Frozen-encoder embeddings, splits, hard negatives (~15 min on an RTX 3090; skips corpus if cached).
python3 prepare.py

# 2. Query-DANN v2 (zero-init L2 adapter, 128-d bottleneck, sanitized discriminator, cosine LR 1e-4 -> 1e-6,
#    tau=0.05, formal preservation mu=0.5, dev eval every 50 steps). gamma sweep selected on dev nDCG@10,
#    plus the no-adversary ablation (gamma=0) under the identical setup.
for g in 0.1 0.5 1.0; do
  python3 query_dann/train.py --run_name v2_dann_g${g} --gamma ${g}
done
python3 query_dann/train.py --run_name v2_infonce_only --gamma 0

# 3. Test-set comparison with 1000-sample paired bootstrap significance tests.
best=$(python3 -c "import json,glob;print(max(glob.glob('runs/v2_dann_*/best/config.json'),key=lambda p:json.load(open(p))['dev']['nDCG@10']).rsplit('/',1)[0])")
python3 eval.py --ckpt "$best" --extra infonce_only=runs/v2_infonce_only/best --split test --bootstrap 1000 --out runs/v2_test_results.json

# First run (spec architecture) is reproducible with:
#   python3 query_dann/train.py --run_name dann_g0.5 --gamma 0.5 --adapter layernorm --discriminator plain --bottleneck 512 \
#     --lr 2e-4 --scheduler none --tau 0.02 --mu 0 --eval_every 45
# tensorboard --logdir runs
