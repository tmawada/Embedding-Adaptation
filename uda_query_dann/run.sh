#!/usr/bin/env bash
# UDA Query-DANN: labelled formal queries (source) -> unlabelled, unpaired informal queries (target).
# Needs the shared embedding cache (../cache, built by ../prepare.py).
set -euo pipefail
cd "$(dirname "$0")"

# gamma = 0 is the source-only baseline; same adapter, schedule, mu=0.5 and 1350-step budget as Query-DANN v2.
for g in 0 0.1 0.5 1.0; do
  python3 train.py --run_name disjoint_g${g} --gamma ${g} --target_split disjoint
done
# Pure DANN objective (no preservation loss).
for g in 0 1.0; do
  python3 train.py --run_name disjoint_g${g}_mu0 --gamma ${g} --mu 0 --target_split disjoint
done
# Overlapping (but unpaired) domains.
for g in 0 1.0; do
  python3 train.py --run_name overlap_g${g} --gamma ${g} --target_split overlap
done

# Strict UDA: checkpoints selected on formal (source) dev queries only.
python3 evaluate.py --reference source_only \
  --run source_only=runs/disjoint_g0/best_source --run uda_g0.1=runs/disjoint_g0.1/best_source \
  --run uda_g0.5=runs/disjoint_g0.5/best_source --run uda_g1.0=runs/disjoint_g1.0/best_source \
  --out runs/test_disjoint_strict.json

# Oracle selection (uses informal dev labels, which strict UDA does not have) - report separately.
python3 evaluate.py --reference source_only \
  --run source_only=runs/disjoint_g0/best_target_oracle --run uda_g0.1=runs/disjoint_g0.1/best_target_oracle \
  --run uda_g0.5=runs/disjoint_g0.5/best_target_oracle --run uda_g1.0=runs/disjoint_g1.0/best_target_oracle \
  --out runs/test_disjoint_oracle.json

# Pure DANN objective (mu = 0), strict selection.
python3 evaluate.py --reference source_only_mu0 \
  --run source_only_mu0=runs/disjoint_g0_mu0/best_source --run uda_g1.0_mu0=runs/disjoint_g1.0_mu0/best_source \
  --out runs/test_disjoint_mu0_strict.json

# Overlapping (but unpaired) domains, strict selection.
python3 evaluate.py --reference source_only \
  --run source_only=runs/overlap_g0/best_source --run uda_g1.0=runs/overlap_g1.0/best_source \
  --out runs/test_overlap_strict.json
