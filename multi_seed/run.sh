#!/usr/bin/env bash
# Seed variance: retrain every trained model with seeds 43-46 (seed 42 checkpoints are reused),
# then report mean ± SD over 5 seeds. Hyperparameters fixed to the dev-selected seed-42 values.
# The TSDAE (task queries) tower stays at its seed-42 checkpoint; only the adapters on top vary.
set -euo pipefail
cd "$(dirname "$0")/.."   # bge_dann/

H=comparison/runs/harmonized
T=$H/sapt_queries/cache
R=multi_seed/runs
SHARED="--lr 3e-5 --lr_min 1e-6 --weight_decay 0.01 --batch_size 64 --eval_every 50 --mixed_precision bf16"

for s in 43 44 45 46; do
  python3 -u query_dann/train.py --output_dir $R/dann_basic --run_name seed$s --gamma 1.0 --epochs 30 --seed $s $SHARED
  python3 -u query_dann/train.py --output_dir $R/dann_linear --run_name seed$s --adapter linear --gamma 1.0 --epochs 30 --seed $s $SHARED
  python3 -u query_distillation/train.py --output_dir $R/distill_score_basic --run_name seed$s --distill score --total_steps 1350 --seed $s $SHARED
  python3 -u query_distillation/train.py --output_dir $R/distill_score_linear --run_name seed$s --distill score --adapter linear --total_steps 1350 --seed $s $SHARED
  python3 -u query_distillation/train.py --output_dir $R/distill_embed_basic --run_name seed$s --distill embed --total_steps 1350 --seed $s $SHARED
  python3 -u query_distillation/train.py --output_dir $R/distill_embed_linear --run_name seed$s --distill embed --adapter linear --total_steps 1350 --seed $s $SHARED
  python3 -u query_dann/train.py --cache_dir $T --output_dir $R/tsdae_dann_linear --run_name seed$s --adapter linear --gamma 1.0 --epochs 30 --seed $s $SHARED
  python3 -u query_distillation/train.py --cache_dir $T --output_dir $R/tsdae_distill_score_linear --run_name seed$s --distill score --adapter linear --total_steps 1350 --seed $s $SHARED
  python3 -u stacked/train.py --cache_dir $T --output_dir $R/stack_tsdae_dann_kd --run_name seed$s --adapter linear --gamma 1.0 --kd_weight 1.0 --epochs 30 --seed $s $SHARED
done

python3 multi_seed/aggregate.py
