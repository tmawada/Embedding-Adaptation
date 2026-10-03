"""Train all 9 models for comprehensive comparison:
1. Query DANN (Linear)
2. Query DANN (Adapter)
3. Query Dist Score (Linear)
4. Query Dist Score (Adapter)
5. TSDAE
6. TSDAE + Query DANN (Linear)
7. TSDAE + Query Dist Score (Linear)
8. TSDAE + Query DANN (Adapter)
9. TSDAE + Query Dist Score (Adapter)

Each model is trained with harmonized hyperparameters for fair comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE


def run_command(cmd: list[str], desc: str, env=None):
    """Run a command and print its output."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    start = time.time()
    result = subprocess.run(cmd, env=env or os.environ.copy(), cwd=str(ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌ FAILED: {desc} (exit code {result.returncode})")
        return False
    else:
        print(f"\n✅ COMPLETED: {desc} ({elapsed:.1f}s)")
        return True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--total_steps", type=int, default=1350, help="Training steps (distillation)")
    ap.add_argument("--epochs", type=int, default=30, help="Training epochs (query DANN)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--gamma", type=float, default=0.1, help="DANN adversarial weight")
    ap.add_argument("--mu", type=float, default=0.5, help="Preservation weight")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--skip_tsdae", action="store_true", help="Skip TSDAE pretraining if already done")
    ap.add_argument("--skip_cache", action="store_true", help="Skip TSDAE cache preparation")
    ap.add_argument("--models", nargs="+", choices=[
        "dann_linear", "dann_adapter",
        "dist_linear", "dist_adapter",
        "tsdae",
        "tsdae_dann_linear", "tsdae_dist_linear",
        "tsdae_dann_adapter", "tsdae_dist_adapter"
    ], help="Specific models to train (default: all)")
    return ap.parse_args()


def train_tsdae(args):
    """Step 1: Train TSDAE base encoder."""
    print("\n" + "="*80)
    print("STEP 1: TSDAE Sentence-Adaptive Pre-Training")
    print("="*80)

    tsdae_dir = ROOT / "tsdae_sapt" / "output" / "bge-m3-sapt-informal-id"

    if args.skip_tsdae and tsdae_dir.exists():
        print(f"⏭️  Skipping TSDAE training (model exists at {tsdae_dir})")
        return True

    cmd = [
        sys.executable, str(ROOT / "tsdae_sapt" / "train_sapt.py"),
        "--epochs", "1",
        "--batch_size", "8",
        "--lr", "3e-5",
        "--seed", str(args.seed)
    ]

    return run_command(cmd, "TSDAE Pre-Training")


def prepare_tsdae_cache(args):
    """Step 2: Prepare TSDAE cached embeddings."""
    print("\n" + "="*80)
    print("STEP 2: Prepare TSDAE Cached Embeddings")
    print("="*80)

    tsdae_cache = ROOT / "tsdae_sapt" / "cache" / "sapt_symmetric"
    # hard_negs.json is the last file prepare_sapt_cache.py writes in --mode symmetric
    cache_file = tsdae_cache / "hard_negs.json"

    if args.skip_cache and cache_file.exists():
        print(f"⏭️  Skipping cache preparation (cache exists at {tsdae_cache})")
        return True

    cmd = [
        sys.executable, str(ROOT / "tsdae_sapt" / "prepare_sapt_cache.py"),
        "--mode", "symmetric",
        "--model_path", str(ROOT / "tsdae_sapt" / "output" / "bge-m3-sapt-informal-id"),
        "--out_cache", str(tsdae_cache),
    ]

    return run_command(cmd, "Prepare TSDAE Cache")


def train_query_dann(args, adapter_type: str, run_name: str, cache_dir: str = None):
    """Train Query DANN with specified adapter."""
    adapter_map = {"linear": "linear", "adapter": "zeroinit"}
    adapter = adapter_map[adapter_type]
    use_adapter = "true"

    cmd = [
        sys.executable, str(ROOT / "query_dann" / "train.py"),
        "--run_name", run_name,
        "--adapter", adapter,
        "--use_adapter", use_adapter,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lr_min", str(args.lr_min),
        "--gamma", str(args.gamma),
        "--mu", str(args.mu),
        "--eval_every", str(args.eval_every),
        "--seed", str(args.seed),
        "--scheduler", "cosine"
    ]

    if cache_dir:
        cmd.extend(["--cache_dir", cache_dir])

    return run_command(cmd, f"Query DANN ({adapter_type})")


def train_query_distill(args, adapter_type: str, run_name: str, cache_dir: str = None):
    """Train Query Distillation (score-level) with specified adapter."""
    adapter_map = {"linear": "linear", "adapter": "zeroinit"}
    adapter = adapter_map[adapter_type]
    use_adapter = "true"

    cmd = [
        sys.executable, str(ROOT / "query_distillation" / "train.py"),
        # query_distillation/train.py defaults to <dir>/runs; evaluation reads runs/
        "--output_dir", str(ROOT / "runs"),
        "--run_name", run_name,
        "--distill", "score",
        "--adapter", adapter,
        "--use_adapter", use_adapter,
        "--total_steps", str(args.total_steps),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lr_min", str(args.lr_min),
        "--mu", str(args.mu),
        "--eval_every", str(args.eval_every),
        "--seed", str(args.seed)
    ]

    if cache_dir:
        cmd.extend(["--cache_dir", cache_dir])

    return run_command(cmd, f"Query Dist Score ({adapter_type})")


def main():
    args = parse_args()

    print("\n" + "="*80)
    print("TRAINING ALL 9 MODELS FOR COMPREHENSIVE COMPARISON")
    print("="*80)
    print(f"Configuration:")
    print(f"  Seed: {args.seed}")
    print(f"  Total steps: {args.total_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr} → {args.lr_min}")
    print(f"  DANN gamma: {args.gamma}")
    print(f"  Preservation mu: {args.mu}")

    models_to_train = args.models or [
        "dann_linear", "dann_adapter",
        "dist_linear", "dist_adapter",
        "tsdae",
        "tsdae_dann_linear", "tsdae_dist_linear",
        "tsdae_dann_adapter", "tsdae_dist_adapter"
    ]

    results = {}
    base_cache = str(ROOT / "cache")
    tsdae_cache = str(ROOT / "tsdae_sapt" / "cache" / "sapt_symmetric")  # SAPT tower, symmetric cache

    # Phase 1: Base models (no TSDAE)
    print("\n" + "="*80)
    print("PHASE 1: BASE MODELS (No TSDAE)")
    print("="*80)

    if "dann_linear" in models_to_train:
        results["dann_linear"] = train_query_dann(
            args, "linear", "query_dann_linear", base_cache
        )

    if "dann_adapter" in models_to_train:
        results["dann_adapter"] = train_query_dann(
            args, "adapter", "query_dann_adapter", base_cache
        )

    if "dist_linear" in models_to_train:
        results["dist_linear"] = train_query_distill(
            args, "linear", "query_dist_linear", base_cache
        )

    if "dist_adapter" in models_to_train:
        results["dist_adapter"] = train_query_distill(
            args, "adapter", "query_dist_adapter", base_cache
        )

    # Phase 2: TSDAE only
    tsdae_needed = any(m.startswith("tsdae") for m in models_to_train)

    if tsdae_needed:
        print("\n" + "="*80)
        print("PHASE 2: TSDAE PRE-TRAINING")
        print("="*80)

        if "tsdae" in models_to_train or any(m.startswith("tsdae_") for m in models_to_train):
            results["tsdae_train"] = train_tsdae(args)

            if results.get("tsdae_train", True):
                results["tsdae_cache"] = prepare_tsdae_cache(args)

    # Phase 3: TSDAE + adaptation methods
    if tsdae_needed and results.get("tsdae_cache", True):
        print("\n" + "="*80)
        print("PHASE 3: TSDAE + ADAPTATION METHODS")
        print("="*80)

        if "tsdae_dann_linear" in models_to_train:
            results["tsdae_dann_linear"] = train_query_dann(
                args, "linear", "tsdae_query_dann_linear", tsdae_cache
            )

        if "tsdae_dist_linear" in models_to_train:
            results["tsdae_dist_linear"] = train_query_distill(
                args, "linear", "tsdae_query_dist_linear", tsdae_cache
            )

        if "tsdae_dann_adapter" in models_to_train:
            results["tsdae_dann_adapter"] = train_query_dann(
                args, "adapter", "tsdae_query_dann_adapter", tsdae_cache
            )

        if "tsdae_dist_adapter" in models_to_train:
            results["tsdae_dist_adapter"] = train_query_distill(
                args, "adapter", "tsdae_query_dist_adapter", tsdae_cache
            )

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    success = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    print(f"\n✅ Successful: {len(success)}/{len(results)}")
    for model in success:
        print(f"   • {model}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for model in failed:
            print(f"   • {model}")

    print("\n" + "="*80)
    print("Next step: Run evaluation script to generate comparison report")
    print(f"Command: python {ROOT / 'evaluate_all_models.py'}")
    print("="*80)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
