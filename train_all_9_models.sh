#!/bin/bash
# Train all 9 models for comprehensive comparison
# Models:
#   1. Query DANN (Linear)
#   2. Query DANN (Adapter)
#   3. Query Dist Score (Linear)
#   4. Query Dist Score (Adapter)
#   5. TSDAE (already trained, zero-shot evaluation)
#   6. TSDAE + Query DANN (Linear)
#   7. TSDAE + Query Dist Score (Linear)
#   8. TSDAE + Query DANN (Adapter)
#   9. TSDAE + Query Dist Score (Adapter)

set -e  # Exit on error

SEED=42
BATCH_SIZE=64
LR=3e-5
LR_MIN=1e-6
GAMMA=0.1
MU=0.5
EVAL_EVERY=50
EPOCHS=30  # For ~1350 steps with batch_size=64

BASE_CACHE="cache"
RUNS_DIR="runs"          # every run (DANN + distillation) lands here; evaluation reads runs/<name>/best
TSDAE_MODEL="tsdae_sapt/output/bge-m3-sapt-informal-id"
# symmetric = SAPT re-encodes queries AND passages, hard negatives re-mined (prepare_sapt_cache.py --mode symmetric)
TSDAE_CACHE="tsdae_sapt/cache/sapt_symmetric"

echo "=========================================="
echo "PHASE 1: BASE MODELS (No TSDAE)"
echo "=========================================="

# 1. Query DANN (Linear)
echo ""
echo "Training: Query DANN (Linear)..."
python3 query_dann/train.py \
    --run_name query_dann_linear \
    --adapter linear \
    --use_adapter true \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lr_min $LR_MIN \
    --gamma $GAMMA \
    --mu $MU \
    --eval_every $EVAL_EVERY \
    --seed $SEED \
    --scheduler cosine \
    --cache_dir $BASE_CACHE

# 2. Query DANN (Adapter)
echo ""
echo "Training: Query DANN (Adapter)..."
python3 query_dann/train.py \
    --run_name query_dann_adapter \
    --adapter zeroinit \
    --use_adapter true \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lr_min $LR_MIN \
    --gamma $GAMMA \
    --mu $MU \
    --eval_every $EVAL_EVERY \
    --seed $SEED \
    --scheduler cosine \
    --cache_dir $BASE_CACHE

# 3. Query Dist Score (Linear)
echo ""
echo "Training: Query Dist Score (Linear)..."
python3 query_distillation/train.py \
    --output_dir $RUNS_DIR \
    --run_name query_dist_linear \
    --distill score \
    --adapter linear \
    --use_adapter true \
    --total_steps 1350 \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lr_min $LR_MIN \
    --mu $MU \
    --eval_every $EVAL_EVERY \
    --seed $SEED \
    --cache_dir $BASE_CACHE

# 4. Query Dist Score (Adapter)
echo ""
echo "Training: Query Dist Score (Adapter)..."
python3 query_distillation/train.py \
    --output_dir $RUNS_DIR \
    --run_name query_dist_adapter \
    --distill score \
    --adapter zeroinit \
    --use_adapter true \
    --total_steps 1350 \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lr_min $LR_MIN \
    --mu $MU \
    --eval_every $EVAL_EVERY \
    --seed $SEED \
    --cache_dir $BASE_CACHE

echo ""
echo "=========================================="
echo "PHASE 2: Prepare TSDAE Cache"
echo "=========================================="

# Check if TSDAE cache is complete (corpus is written before hard_negs.json, which is the last step)
if [ ! -f "$TSDAE_CACHE/corpus_emb.npy" ] || [ ! -f "$TSDAE_CACHE/hard_negs.json" ]; then
    echo "Preparing TSDAE cache (mode=symmetric) -> $TSDAE_CACHE ..."
    python3 tsdae_sapt/prepare_sapt_cache.py \
        --mode symmetric \
        --model_path "$TSDAE_MODEL" \
        --out_cache "$TSDAE_CACHE"
else
    echo "TSDAE cache already exists at $TSDAE_CACHE, skipping..."
fi

echo ""
echo "=========================================="
echo "PHASE 3: TSDAE + ADAPTATION METHODS"
echo "=========================================="

# 6. TSDAE + Query DANN (Linear)
echo ""
echo "Training: TSDAE + Query DANN (Linear)..."
python3 query_dann/train.py \
    --run_name tsdae_query_dann_linear \
    --adapter linear \
    --use_adapter true \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lr_min $LR_MIN \
    --gamma $GAMMA \
    --mu $MU \
    --eval_every $EVAL_EVERY \
    --seed $SEED \
    --scheduler cosine \
    --cache_dir $TSDAE_CACHE

# 7. TSDAE + Query Dist Score (Linear)
echo ""
echo "Training: TSDAE + Query Dist Score (Linear)..."
python3 query_distillation/train.py \
    --output_dir $RUNS_DIR \
    --run_name tsdae_query_dist_linear \
    --distill score \
    --adapter linear \
    --use_adapter true \
    --total_steps 1350 \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lr_min $LR_MIN \
    --mu $MU \
    --eval_every $EVAL_EVERY \
    --seed $SEED \
    --cache_dir $TSDAE_CACHE

# 8. TSDAE + Query DANN (Adapter)
echo ""
echo "Training: TSDAE + Query DANN (Adapter)..."
python3 query_dann/train.py \
    --run_name tsdae_query_dann_adapter \
    --adapter zeroinit \
    --use_adapter true \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lr_min $LR_MIN \
    --gamma $GAMMA \
    --mu $MU \
    --eval_every $EVAL_EVERY \
    --seed $SEED \
    --scheduler cosine \
    --cache_dir $TSDAE_CACHE

# 9. TSDAE + Query Dist Score (Adapter)
echo ""
echo "Training: TSDAE + Query Dist Score (Adapter)..."
python3 query_distillation/train.py \
    --output_dir $RUNS_DIR \
    --run_name tsdae_query_dist_adapter \
    --distill score \
    --adapter zeroinit \
    --use_adapter true \
    --total_steps 1350 \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lr_min $LR_MIN \
    --mu $MU \
    --eval_every $EVAL_EVERY \
    --seed $SEED \
    --cache_dir $TSDAE_CACHE

echo ""
echo "=========================================="
echo "ALL MODELS TRAINED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Next step: Run evaluation to generate comparison report"
echo "Command: python3 evaluate_all_models.py"
echo ""
