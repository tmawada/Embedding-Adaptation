#!/usr/bin/env bash
# Stacked: TSDAE-SAPT (task queries) tower + Query-DANN v2 + score distillation, trained as ONE linear adapter
# with a joint loss. Harmonized protocol; lambda (kd_weight) picked on dev nDCG@10.
set -euo pipefail
cd "$(dirname "$0")/.."   # bge_dann/

H=comparison/runs/harmonized; A=$H/ablation_adapter; S=stacked/runs
SHARED="--adapter linear --gamma 1.0 --epochs 30 --lr 3e-5 --lr_min 1e-6 --weight_decay 0.01 --batch_size 64 --eval_every 50 --seed 42 --mixed_precision bf16"

# 0. Sanity: kd_weight 0 must reproduce Query-DANN (linear) exactly.
python3 -u stacked/train.py --run_name base/kd0 --kd_weight 0 $SHARED

# 1. DANN + KD on the base tower (isolates TSDAE) and on the TSDAE tower (the stack).
for lam in 0.5 1.0; do
  python3 -u stacked/train.py --run_name base/kd$lam --kd_weight $lam $SHARED
  python3 -u stacked/train.py --run_name sapt_queries/kd$lam --kd_weight $lam --cache_dir $H/sapt_queries/cache $SHARED
done

best() { python3 -c "import json;ps=['$S/$1/kd0.5/best','$S/$1/kd1.0/best'];print(max(ps,key=lambda p:json.load(open(p+'/config.json'))['dev']['nDCG@10']))"; }
BASE_KD=$(best base); STACK=$(best sapt_queries)

# 2. Table (formal/informal metrics + nDCG gap).
python3 comparison/report.py --out_dir $S --report_name stack_tsdae_dann_kd \
  --skip_default_adapters --skip_sapt_stacks --sapt_variants queries \
  --extra "Query-DANN v2 (linear adapter)=$A/dann/linear/best" \
  --extra "Query distillation score (linear adapter)=$A/distill/linear/best" \
  --extra "Query-DANN v2 + distillation score (linear adapter)=$BASE_KD" \
  --extra_tower "TSDAE-SAPT (task queries) + Query-DANN v2 (linear adapter)=queries=$A/sapt_queries/dann_linear/best" \
  --extra_tower "TSDAE-SAPT (task queries) + Query distillation score (linear adapter)=queries=$A/sapt_queries/distill_linear/best" \
  --extra_tower "TSDAE-SAPT (task queries) + Query-DANN v2 + distillation score (linear adapter)=queries=$STACK"

# 3. Head-to-head significance tests.
python3 stacked/head_to_head.py --stack "$STACK" --base_kd "$BASE_KD"
