#!/bin/bash
# Complete pipeline: Train all 9 models and generate comparison report

set -e  # Exit on error

# Same cache the training script prepared (must match train_all_9_models.sh)
TSDAE_MODEL="tsdae_sapt/output/bge-m3-sapt-informal-id"
TSDAE_CACHE="tsdae_sapt/cache/sapt_symmetric"

echo "🚀 Starting Complete Model Comparison Pipeline"
echo "=========================================="

# Step 1: Train all models
echo ""
echo "📚 PHASE 1: TRAINING ALL 9 MODELS"
echo "=========================================="
./train_all_9_models.sh

# Step 2: Prepare TSDAE cache if needed (train_all_9_models.sh already did this; kept as a safety net)
echo ""
echo "📦 PHASE 2: PREPARING CACHES"
echo "=========================================="
if [ ! -f "$TSDAE_CACHE/corpus_emb.npy" ] || [ ! -f "$TSDAE_CACHE/hard_negs.json" ]; then
    echo "Preparing TSDAE cache (mode=symmetric) -> $TSDAE_CACHE ..."
    python3 tsdae_sapt/prepare_sapt_cache.py \
        --mode symmetric \
        --model_path "$TSDAE_MODEL" \
        --out_cache "$TSDAE_CACHE"
else
    echo "TSDAE cache already exists at $TSDAE_CACHE"
fi

# Step 3: Evaluate all models and generate report
echo ""
echo "📊 PHASE 3: EVALUATION AND REPORT GENERATION"
echo "=========================================="
echo "Running evaluation on test set..."
python3 evaluate_comparison.py --split test --output comparison_results.json \
    --tsdae_cache_dir "$TSDAE_CACHE"

echo ""
echo "📈 PHASE 4: GENERATING DASHBOARD"
echo "=========================================="
# Copy results to a location the dashboard can access
cp comparison_results.json comparison_results_latest.json

echo ""
echo "✅ PIPELINE COMPLETE!"
echo "=========================================="
echo "📁 Results available in:"
echo "   - comparison_results.json (detailed JSON)"
echo "   - comparison_results_latest.json (for dashboard)"
echo "   - model_comparison_dashboard.html (interactive dashboard)"
echo ""
echo "🌐 To view results:"
echo "   Open model_comparison_dashboard.html in your browser"
echo ""
echo "📋 Key files trained:"
echo "   1. query_dann_linear"
echo "   2. query_dann_adapter"
echo "   3. query_dist_linear"
echo "   4. query_dist_adapter"
echo "   5. TSDAE (zero-shot)"
echo "   6. tsdae_query_dann_linear"
echo "   7. tsdae_query_dist_linear"
echo "   8. tsdae_query_dann_adapter"
echo "   9. tsdae_query_dist_adapter"
echo ""
