"""Evaluate all 9 trained models and generate comprehensive comparison report.

Compares:
1. Query DANN (Linear)
2. Query DANN (Adapter)
3. Query Dist Score (Linear)
4. Query Dist Score (Adapter)
5. TSDAE (zero-shot)
6. TSDAE + Query DANN (Linear)
7. TSDAE + Query Dist Score (Linear)
8. TSDAE + Query DANN (Adapter)
9. TSDAE + Query Dist Score (Adapter)

Metrics: Recall@100, MRR@10, nDCG@10 for both formal and informal queries.
Main comparison: formal vs informal gap using nDCG@10.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
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
        print(f"⚠️  Model not found at {best_dir}")
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


def evaluate_model(model: torch.nn.Module, evaluator: Evaluator,
                   split_name: str) -> Dict[str, Dict[str, float]]:
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
            # Bootstrap test for gap
            boot = paired_bootstrap(per_query_form[metric], per_query_inf[metric])
            gap_stats[metric] = boot

    return {
        "informal": metrics_inf,
        "formal": metrics_form,
        "gap": gap,
        "gap_stats": gap_stats,
        "per_query_informal": {k: v.tolist() for k, v in per_query_inf.items()},
        "per_query_formal": {k: v.tolist() for k, v in per_query_form.items()}
    }


def evaluate_baseline(evaluator: Evaluator, split_name: str,
                      is_tsdae: bool = False) -> Dict[str, Dict[str, float]]:
    """Evaluate baseline (no adaptation)."""
    with torch.no_grad():
        # Informal queries (base embeddings)
        z_inf = evaluator.informal()
        metrics_inf = evaluator.run(z_inf)
        per_query_inf = evaluator.run_per_query(z_inf)

        # Formal queries (base embeddings)
        z_form = evaluator.formal()
        metrics_form = evaluator.run(z_form)
        per_query_form = evaluator.run_per_query(z_form)

        # Compute gap
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
        "gap_stats": gap_stats,
        "per_query_informal": {k: v.tolist() for k, v in per_query_inf.items()},
        "per_query_formal": {k: v.tolist() for k, v in per_query_form.items()}
    }


def format_metric_row(name: str, metrics_inf: dict, metrics_form: dict, gap: dict) -> str:
    """Format a single row for the results table."""
    return (f"{name:30s} | "
            f"Inf: R@100={metrics_inf['Recall@100']:.4f} MRR@10={metrics_inf['MRR@10']:.4f} nDCG@10={metrics_inf['nDCG@10']:.4f} | "
            f"Form: R@100={metrics_form['Recall@100']:.4f} MRR@10={metrics_form['MRR@10']:.4f} nDCG@10={metrics_form['nDCG@10']:.4f} | "
            f"Gap: {gap['nDCG@10']:+.4f}")


def generate_markdown_report(all_results: Dict, split: str, output_path: Path):
    """Generate markdown comparison report."""

    lines = [
        f"# Model Comparison Report - {split.upper()} Split",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "Comprehensive comparison of 9 models on MIRACL Indonesian retrieval:",
        "- **Base Models**: Query DANN and Query Distillation with Linear/Adapter variants",
        "- **TSDAE Models**: TSDAE pre-training + Query DANN/Distillation",
        "- **Metrics**: Recall@100, MRR@10, nDCG@10",
        "- **Main Focus**: Formal vs Informal gap (nDCG@10)",
        "",
        "## Results Table",
        "",
        "| Model | Informal nDCG@10 | Formal nDCG@10 | Gap (F-I) | Informal MRR@10 | Formal MRR@10 | Informal R@100 | Formal R@100 |",
        "|-------|------------------|----------------|-----------|-----------------|---------------|----------------|--------------|"
    ]

    # Order models for display
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
            continue

        res = all_results[key]
        inf = res["informal"]
        form = res["formal"]
        gap = res["gap"]

        lines.append(
            f"| {display_name:25s} | "
            f"{inf['nDCG@10']:6.4f} | "
            f"{form['nDCG@10']:6.4f} | "
            f"{gap['nDCG@10']:+6.4f} | "
            f"{inf['MRR@10']:6.4f} | "
            f"{form['MRR@10']:6.4f} | "
            f"{inf['Recall@100']:6.4f} | "
            f"{form['Recall@100']:6.4f} |"
        )

    lines.extend([
        "",
        "## Gap Analysis (nDCG@10)",
        "",
        "Positive gap = Formal queries outperform Informal queries",
        "",
        "### Gap Reduction from Baseline",
        ""
    ])

    if "baseline" in all_results:
        baseline_gap = all_results["baseline"]["gap"]["nDCG@10"]
        lines.append(f"**Baseline Gap**: {baseline_gap:+.4f}\n")

        for display_name, key in model_order[1:]:  # Skip baseline
            if key not in all_results:
                continue

            gap = all_results[key]["gap"]["nDCG@10"]
            reduction = baseline_gap - gap
            reduction_pct = (reduction / abs(baseline_gap)) * 100 if baseline_gap != 0 else 0

            lines.append(
                f"- **{display_name}**: Gap={gap:+.4f}, "
                f"Reduction={reduction:+.4f} ({reduction_pct:+.1f}%)"
            )

    lines.extend([
        "",
        "## Statistical Significance",
        "",
        "Paired bootstrap tests (1000 samples) for gap differences:",
        ""
    ])

    for display_name, key in model_order:
        if key not in all_results:
            continue

        gap_stats = all_results[key]["gap_stats"]["nDCG@10"]
        lines.append(
            f"### {display_name}\n"
            f"- Gap: {gap_stats['diff']:+.4f}\n"
            f"- 95% CI: [{gap_stats['ci95_low']:+.4f}, {gap_stats['ci95_high']:+.4f}]\n"
            f"- p-value: {gap_stats['p']:.4f}\n"
        )

    lines.extend([
        "",
        "## Best Performers",
        ""
    ])

    # Find best models
    best_inf = max(all_results.items(), key=lambda x: x[1]["informal"]["nDCG@10"])
    best_form = max(all_results.items(), key=lambda x: x[1]["formal"]["nDCG@10"])
    smallest_gap = min(all_results.items(), key=lambda x: abs(x[1]["gap"]["nDCG@10"]))

    lines.extend([
        f"- **Best Informal nDCG@10**: {best_inf[0]} ({best_inf[1]['informal']['nDCG@10']:.4f})",
        f"- **Best Formal nDCG@10**: {best_form[0]} ({best_form[1]['formal']['nDCG@10']:.4f})",
        f"- **Smallest Gap**: {smallest_gap[0]} ({smallest_gap[1]['gap']['nDCG@10']:+.4f})",
        ""
    ])

    # Write report
    output_path.write_text("\n".join(lines))
    print(f"\n✅ Report saved to: {output_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--cache_dir", default=str(HERE / "cache"))
    ap.add_argument("--data_dir", default=str(HERE / "data"))
    ap.add_argument("--runs_dir", default=str(HERE / "runs"))
    ap.add_argument("--tsdae_cache_dir", default=str(HERE / "tsdae_sapt" / "cache" / "sapt_symmetric"),
                    help="Cache prepared by tsdae_sapt/prepare_sapt_cache.py (models 5-9)")
    ap.add_argument("--output", default=str(HERE / "comparison_report.md"))
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*80}")
    print(f"EVALUATING ALL MODELS ON {args.split.upper()} SPLIT")
    print(f"{'='*80}\n")

    # Load data
    base_store = EmbeddingStore(args.cache_dir, device=device)
    with open(Path(args.cache_dir) / "splits.json") as f:
        splits = json.load(f)

    # Check for TSDAE cache (models 5-9 use their own query + passage embeddings)
    tsdae_cache_dir = args.tsdae_cache_dir
    tsdae_available = (Path(tsdae_cache_dir) / "formal_emb.npy").exists() \
        and (Path(tsdae_cache_dir) / "corpus_emb.npy").exists() \
        and (Path(tsdae_cache_dir) / "hard_negs.json").exists()

    if tsdae_available:
        tsdae_store = EmbeddingStore(tsdae_cache_dir, device=device)
    else:
        print(f"⚠️  TSDAE cache not found at {tsdae_cache_dir}, TSDAE models will be skipped")
        print("    Run: python3 tsdae_sapt/prepare_sapt_cache.py --mode symmetric "
              f"--out_cache {tsdae_cache_dir}\n")
        tsdae_store = None

    evaluator = Evaluator(base_store, splits[args.split], args.data_dir)
    if tsdae_store:
        tsdae_evaluator = Evaluator(tsdae_store, splits[args.split], args.data_dir)

    all_results = {}

    # 1. Baseline (no adaptation)
    print("Evaluating: Baseline (BGE-M3)...")
    all_results["baseline"] = evaluate_baseline(evaluator, args.split, is_tsdae=False)
    print(f"  Informal nDCG@10: {all_results['baseline']['informal']['nDCG@10']:.4f}")
    print(f"  Formal nDCG@10: {all_results['baseline']['formal']['nDCG@10']:.4f}")
    print(f"  Gap: {all_results['baseline']['gap']['nDCG@10']:+.4f}\n")

    # 2-5. Base adaptation models
    base_models = {
        "dann_linear": "query_dann_linear",
        "dann_adapter": "query_dann_adapter",
        "dist_linear": "query_dist_linear",
        "dist_adapter": "query_dist_adapter"
    }

    for key, run_name in base_models.items():
        run_dir = Path(args.runs_dir) / run_name
        print(f"Evaluating: {key}...")

        model = load_model(run_dir, device)
        if model is None:
            print(f"  ❌ Skipped (model not found)\n")
            continue

        all_results[key] = evaluate_model(model, evaluator, args.split)
        print(f"  Informal nDCG@10: {all_results[key]['informal']['nDCG@10']:.4f}")
        print(f"  Formal nDCG@10: {all_results[key]['formal']['nDCG@10']:.4f}")
        print(f"  Gap: {all_results[key]['gap']['nDCG@10']:+.4f}\n")

    # 6. TSDAE only (zero-shot)
    if tsdae_store:
        print("Evaluating: TSDAE Only...")
        all_results["tsdae"] = evaluate_baseline(tsdae_evaluator, args.split, is_tsdae=True)
        print(f"  Informal nDCG@10: {all_results['tsdae']['informal']['nDCG@10']:.4f}")
        print(f"  Formal nDCG@10: {all_results['tsdae']['formal']['nDCG@10']:.4f}")
        print(f"  Gap: {all_results['tsdae']['gap']['nDCG@10']:+.4f}\n")

        # 7-9. TSDAE + adaptation models
        tsdae_models = {
            "tsdae_dann_linear": "tsdae_query_dann_linear",
            "tsdae_dann_adapter": "tsdae_query_dann_adapter",
            "tsdae_dist_linear": "tsdae_query_dist_linear",
            "tsdae_dist_adapter": "tsdae_query_dist_adapter"
        }

        for key, run_name in tsdae_models.items():
            run_dir = Path(args.runs_dir) / run_name
            print(f"Evaluating: {key}...")

            model = load_model(run_dir, device)
            if model is None:
                print(f"  ❌ Skipped (model not found)\n")
                continue

            all_results[key] = evaluate_model(model, tsdae_evaluator, args.split)
            print(f"  Informal nDCG@10: {all_results[key]['informal']['nDCG@10']:.4f}")
            print(f"  Formal nDCG@10: {all_results[key]['formal']['nDCG@10']:.4f}")
            print(f"  Gap: {all_results[key]['gap']['nDCG@10']:+.4f}\n")

    # Generate report
    print(f"{'='*80}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*80}\n")

    # Save raw results
    results_json = Path(args.output).with_suffix(".json")
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Raw results saved to: {results_json}")

    # Generate markdown report
    generate_markdown_report(all_results, args.split, Path(args.output))

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
