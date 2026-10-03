"""Evaluate all 9 trained models and generate comprehensive comparison report.

Compares all models on formal/informal queries with Recall, MRR, nDCG metrics.
Main focus: formal vs informal gap using nDCG@10.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from dataset import EmbeddingStore
from eval import Evaluator, paired_bootstrap
from model import QueryDANN, EMB_DIM, ADAPTERS
import torch.nn.functional as F


class AdapterHead(torch.nn.Module):
    """Adapter-only head for distillation models."""

    def __init__(self, adapter: str, bottleneck: int, dropout: float):
        super().__init__()
        self.adapter = ADAPTERS[adapter](EMB_DIM, bottleneck, dropout)

    def adapt(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.adapter(z).float(), dim=-1)


def load_model(run_dir: Path, device: torch.device) -> Optional[torch.nn.Module]:
    """Load a trained model checkpoint."""
    best_dir = run_dir / "best"
    config_path = best_dir / "config.json"
    adapter_path = best_dir / "adapter.pt"

    if not config_path.exists() or not adapter_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    adapter_type = config.get("adapter", "zeroinit")
    bottleneck = config.get("bottleneck", 128)
    adapter_dropout = config.get("adapter_dropout", 0.1)
    disc_dropout = config.get("disc_dropout", 0.2)
    discriminator_type = config.get("discriminator", "sanitized")

    # Determine if this is a DANN or distillation model
    if "discriminator" in config and (best_dir / "discriminator.pt").exists():
        # Query DANN model
        model = QueryDANN(bottleneck, adapter_dropout, disc_dropout,
                         adapter_type, discriminator_type).to(device)
        model.adapter.load_state_dict(torch.load(adapter_path, map_location=device))
        model.discriminator.load_state_dict(torch.load(best_dir / "discriminator.pt", map_location=device))
    else:
        # Distillation or linear model
        model = AdapterHead(adapter_type, bottleneck, adapter_dropout).to(device)
        model.adapter.load_state_dict(torch.load(adapter_path, map_location=device))

    model.eval()
    return model


def evaluate_model(model: torch.nn.Module, evaluator: Evaluator) -> Dict:
    """Evaluate model on both formal and informal queries."""
    with torch.no_grad():
        # Informal queries (adapted)
        z_inf = evaluator.adapted(model, "informal")
        metrics_inf = evaluator.run(z_inf)
        per_query_inf = evaluator.run_per_query(z_inf)

        # Formal queries (adapted)
        z_form = evaluator.adapted(model, "formal")
        metrics_form = evaluator.run(z_form)
        per_query_form = evaluator.run_per_query(z_form)

        # Compute gap (formal - informal)
        gap = {}
        gap_stats = {}
        for metric in ["nDCG@10", "MRR@10", "Recall@100"]:
            gap[metric] = metrics_form[metric] - metrics_inf[metric]
            boot = paired_bootstrap(per_query_form[metric], per_query_inf[metric])
            gap_stats[metric] = boot

    return {
        "informal": metrics_inf,
        "formal": metrics_form,
        "gap": gap,
        "gap_stats": gap_stats
    }


def evaluate_baseline(evaluator: Evaluator) -> Dict:
    """Evaluate baseline (no adaptation)."""
    with torch.no_grad():
        z_inf = evaluator.informal()
        metrics_inf = evaluator.run(z_inf)
        per_query_inf = evaluator.run_per_query(z_inf)

        z_form = evaluator.formal()
        metrics_form = evaluator.run(z_form)
        per_query_form = evaluator.run_per_query(z_form)

        gap = {}
        gap_stats = {}
        for metric in ["nDCG@10", "MRR@10", "Recall@100"]:
            gap[metric] = metrics_form[metric] - metrics_inf[metric]
            boot = paired_bootstrap(per_query_form[metric], per_query_inf[metric])
            gap_stats[metric] = boot

    return {
        "informal": metrics_inf,
        "formal": metrics_form,
        "gap": gap,
        "gap_stats": gap_stats
    }


def print_results_table(all_results: Dict):
    """Print formatted results table."""
    print("\n" + "="*120)
    print("RESULTS TABLE")
    print("="*120)
    print(f"{'Model':<30} | {'Inf nDCG@10':>11} | {'Form nDCG@10':>11} | {'Gap':>8} | {'Inf MRR@10':>11} | {'Inf R@100':>11}")
    print("-"*120)

    model_order = [
        ("Baseline (BGE-M3)", "baseline"),
        ("Query DANN (Linear)", "dann_linear"),
        ("Query DANN (Adapter)", "dann_adapter"),
        ("Query Dist (Linear)", "dist_linear"),
        ("Query Dist (Adapter)", "dist_adapter"),
        ("TSDAE Only", "tsdae"),
        ("TSDAE + DANN (Linear)", "tsdae_dann_linear"),
        ("TSDAE + DANN (Adapter)", "tsdae_dann_adapter"),
        ("TSDAE + Dist (Linear)", "tsdae_dist_linear"),
        ("TSDAE + Dist (Adapter)", "tsdae_dist_adapter"),
    ]

    for display_name, key in model_order:
        if key not in all_results:
            print(f"{display_name:<30} | {'MISSING':>11} | {'MISSING':>11} | {'':>8} | {'':>11} | {'':>11}")
            continue

        res = all_results[key]
        inf = res["informal"]
        form = res["formal"]
        gap = res["gap"]

        print(f"{display_name:<30} | {inf['nDCG@10']:>11.4f} | {form['nDCG@10']:>11.4f} | "
              f"{gap['nDCG@10']:>+8.4f} | {inf['MRR@10']:>11.4f} | {inf['Recall@100']:>11.4f}")

    print("="*120)


def print_gap_analysis(all_results: Dict):
    """Print gap analysis."""
    print("\n" + "="*80)
    print("GAP ANALYSIS (nDCG@10)")
    print("="*80)

    if "baseline" not in all_results:
        print("Baseline not found, cannot compute gap reduction")
        return

    baseline_gap = all_results["baseline"]["gap"]["nDCG@10"]
    print(f"\nBaseline Gap (Formal - Informal): {baseline_gap:+.4f}\n")
    print(f"{'Model':<35} {'Gap':>8} {'Reduction':>10} {'Reduction %':>12}")
    print("-"*80)

    for display_name, key in [
        ("Query DANN (Linear)", "dann_linear"),
        ("Query DANN (Adapter)", "dann_adapter"),
        ("Query Dist (Linear)", "dist_linear"),
        ("Query Dist (Adapter)", "dist_adapter"),
        ("TSDAE Only", "tsdae"),
        ("TSDAE + DANN (Linear)", "tsdae_dann_linear"),
        ("TSDAE + DANN (Adapter)", "tsdae_dann_adapter"),
        ("TSDAE + Dist (Linear)", "tsdae_dist_linear"),
        ("TSDAE + Dist (Adapter)", "tsdae_dist_adapter"),
    ]:
        if key not in all_results:
            continue

        gap = all_results[key]["gap"]["nDCG@10"]
        reduction = baseline_gap - gap
        reduction_pct = (reduction / abs(baseline_gap)) * 100 if baseline_gap != 0 else 0

        print(f"{display_name:<35} {gap:>+8.4f} {reduction:>+10.4f} {reduction_pct:>11.1f}%")

    print("="*80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--cache_dir", default=str(HERE / "cache"))
    ap.add_argument("--data_dir", default=str(HERE / "data"))
    ap.add_argument("--runs_dir", default=str(HERE / "runs"))
    ap.add_argument("--tsdae_cache_dir", default=str(HERE / "tsdae_sapt" / "cache" / "sapt_symmetric"),
                    help="Cache prepared by tsdae_sapt/prepare_sapt_cache.py (models 5-9)")
    ap.add_argument("--output", default=str(HERE / "comparison_results.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*80}")
    print(f"EVALUATING ALL MODELS ON {args.split.upper()} SPLIT")
    print(f"{'='*80}\n")

    # Load data
    base_store = EmbeddingStore(args.cache_dir, device=device)
    with open(Path(args.cache_dir) / "splits.json") as f:
        splits = json.load(f)

    base_evaluator = Evaluator(base_store, splits[args.split], args.data_dir)

    # TSDAE tower cache (models 5-9): its own query AND passage embeddings, mined negatives
    tsdae_cache_ok = (Path(args.tsdae_cache_dir) / "formal_emb.npy").exists() \
        and (Path(args.tsdae_cache_dir) / "corpus_emb.npy").exists() \
        and (Path(args.tsdae_cache_dir) / "hard_negs.json").exists()
    if tsdae_cache_ok:
        tsdae_store = EmbeddingStore(args.tsdae_cache_dir, device=device)
        tsdae_evaluator = Evaluator(tsdae_store, splits[args.split], args.data_dir)
    else:
        tsdae_evaluator = None
        print(f"⚠️  TSDAE cache not found at {args.tsdae_cache_dir}, models 5-9 will be skipped")
        print("    Run: python3 tsdae_sapt/prepare_sapt_cache.py --mode symmetric "
              f"--out_cache {args.tsdae_cache_dir}\n")

    all_results = {}

    # 1. Baseline
    print("Evaluating: Baseline (BGE-M3)...")
    all_results["baseline"] = evaluate_baseline(base_evaluator)
    print(f"  ✓ Informal nDCG@10: {all_results['baseline']['informal']['nDCG@10']:.4f}, "
          f"Gap: {all_results['baseline']['gap']['nDCG@10']:+.4f}\n")

    # 2-5. Base adaptation models
    base_models = {
        "dann_linear": ("query_dann_linear", "Query DANN (Linear)"),
        "dann_adapter": ("query_dann_adapter", "Query DANN (Adapter)"),
        "dist_linear": ("query_dist_linear", "Query Dist (Linear)"),
        "dist_adapter": ("query_dist_adapter", "Query Dist (Adapter)")
    }

    for key, (run_name, display_name) in base_models.items():
        run_dir = Path(args.runs_dir) / run_name
        print(f"Evaluating: {display_name}...")

        model = load_model(run_dir, device)
        if model is None:
            print(f"  ❌ Skipped (model not found at {run_dir})\n")
            continue

        all_results[key] = evaluate_model(model, base_evaluator)
        print(f"  ✓ Informal nDCG@10: {all_results[key]['informal']['nDCG@10']:.4f}, "
              f"Gap: {all_results[key]['gap']['nDCG@10']:+.4f}\n")

    # 5-9. TSDAE tower models (own query + passage embeddings)
    if tsdae_evaluator is not None:
        print("Evaluating: TSDAE Only (zero-shot)...")
        all_results["tsdae"] = evaluate_baseline(tsdae_evaluator)
        print(f"  ✓ Informal nDCG@10: {all_results['tsdae']['informal']['nDCG@10']:.4f}, "
              f"Gap: {all_results['tsdae']['gap']['nDCG@10']:+.4f}\n")

        tsdae_models = {
            "tsdae_dann_linear": ("tsdae_query_dann_linear", "TSDAE + DANN (Linear)"),
            "tsdae_dann_adapter": ("tsdae_query_dann_adapter", "TSDAE + DANN (Adapter)"),
            "tsdae_dist_linear": ("tsdae_query_dist_linear", "TSDAE + Dist (Linear)"),
            "tsdae_dist_adapter": ("tsdae_query_dist_adapter", "TSDAE + Dist (Adapter)")
        }

        for key, (run_name, display_name) in tsdae_models.items():
            run_dir = Path(args.runs_dir) / run_name
            print(f"Evaluating: {display_name}...")

            model = load_model(run_dir, device)
            if model is None:
                print(f"  ❌ Skipped (model not found at {run_dir})\n")
                continue

            all_results[key] = evaluate_model(model, tsdae_evaluator)
            print(f"  ✓ Informal nDCG@10: {all_results[key]['informal']['nDCG@10']:.4f}, "
                  f"Gap: {all_results[key]['gap']['nDCG@10']:+.4f}\n")
    else:
        print("⚠️  TSDAE-based models (5-9) skipped: no TSDAE cache\n")

    # Print results
    print_results_table(all_results)
    print_gap_analysis(all_results)

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to: {args.output}\n")
    print(f"{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
