#!/usr/bin/env python3
"""Evaluate selected 7 models for comparison.

Models:
1. BGE-M3 base (baseline)
2. BGE-M3 + LLM Normalization
3. Query DANN
4. Query Distillation (Linear)
5. Query DANN + Adapter (Zero Init)
6. TSDAE
7. TSDAE + Meaning
8. Query InfoNCE
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

from preprocessing.dataset import EmbeddingStore
from methods.eval import Evaluator, paired_bootstrap
from methods.model import QueryDANN, EMB_DIM, ADAPTERS
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


def evaluate_model(model: torch.nn.Module, evaluator: Evaluator,
                   with_std: bool = False) -> Dict:
    """Evaluate model on both formal and informal queries."""
    with torch.no_grad():
        # Informal queries (adapted)
        z_inf = evaluator.adapted(model, "informal")
        per_query_inf = evaluator.run_per_query(z_inf)

        # Formal queries (adapted)
        z_form = evaluator.adapted(model, "formal")
        per_query_form = evaluator.run_per_query(z_form)

        # Compute metrics with or without std
        if with_std:
            metrics_inf = {m: {"mean": float(v.mean()), "std": float(v.std(ddof=0))}
                          for m, v in per_query_inf.items()}
            metrics_form = {m: {"mean": float(v.mean()), "std": float(v.std(ddof=0))}
                           for m, v in per_query_form.items()}
        else:
            metrics_inf = {m: float(v.mean()) for m, v in per_query_inf.items()}
            metrics_form = {m: float(v.mean()) for m, v in per_query_form.items()}

        # Compute gap (formal - informal)
        gap = {}
        gap_stats = {}
        for metric in ["nDCG@10", "MRR@10", "Recall@100"]:
            if with_std:
                gap[metric] = {"mean": metrics_form[metric]["mean"] - metrics_inf[metric]["mean"],
                              "std": float(np.sqrt(per_query_form[metric].var(ddof=0) + per_query_inf[metric].var(ddof=0)))}
            else:
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


def evaluate_baseline(evaluator: Evaluator,
                      is_tsdae: bool = False,
                      with_std: bool = False) -> Dict:
    """Evaluate baseline (no adaptation)."""
    with torch.no_grad():
        # Informal queries (base embeddings)
        z_inf = evaluator.informal()
        per_query_inf = evaluator.run_per_query(z_inf)

        # Formal queries (base embeddings)
        z_form = evaluator.formal()
        per_query_form = evaluator.run_per_query(z_form)

        # Compute metrics with or without std
        if with_std:
            metrics_inf = {m: {"mean": float(v.mean()), "std": float(v.std(ddof=0))}
                          for m, v in per_query_inf.items()}
            metrics_form = {m: {"mean": float(v.mean()), "std": float(v.std(ddof=0))}
                           for m, v in per_query_form.items()}
        else:
            metrics_inf = {m: float(v.mean()) for m, v in per_query_inf.items()}
            metrics_form = {m: float(v.mean()) for m, v in per_query_form.items()}

        # Compute gap
        gap = {}
        gap_stats = {}
        for metric in ["nDCG@10", "MRR@10", "Recall@100"]:
            if with_std:
                gap[metric] = {"mean": metrics_form[metric]["mean"] - metrics_inf[metric]["mean"],
                              "std": float(np.sqrt(per_query_form[metric].var(ddof=0) + per_query_inf[metric].var(ddof=0)))}
            else:
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


def load_llm_normalization_results(cache_dir: str, data_dir: str, split: str,
                                   with_std: bool = False) -> Dict:
    """Load LLM normalization results from llm_rewrite."""
    from preprocessing.dataset import EmbeddingStore, load_qrels
    import torch

    # First, try to get results from the JSON file if split is test
    if split == "test":
        llm_results_path = Path(cache_dir) / ".." / "llm_rewrite" / "runs" / "test_results.json"
        if not llm_results_path.exists():
            # Try alternative path
            llm_results_path = HERE / "llm_rewrite" / "runs" / "test_results.json"

        if llm_results_path.exists():
            with open(llm_results_path) as f:
                llm_data = json.load(f)

            # Extract the LLM normalization result
            for entry in llm_data.get("results", {}):
                if "LLM-normalised query" in entry:
                    value = llm_data["results"][entry]
                    if with_std:
                        informal = {m: {"mean": value[m], "std": 0.0} for m in ["nDCG@10", "MRR@10", "Recall@100"]}
                        formal = {m: {"mean": value[m], "std": 0.0} for m in ["nDCG@10", "MRR@10", "Recall@100"]}
                        gap = {m: {"mean": 0.0, "std": 0.0} for m in ["nDCG@10", "MRR@10", "Recall@100"]}
                        gap_stats = {m: {"diff": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p": 1.0} for m in ["nDCG@10", "MRR@10", "Recall@100"]}
                    else:
                        informal = {m: value[m] for m in ["nDCG@10", "MRR@10", "Recall@100"]}
                        formal = {m: value[m] for m in ["nDCG@10", "MRR@10", "Recall@100"]}
                        gap = {m: 0.0 for m in ["nDCG@10", "MRR@10", "Recall@100"]}
                        gap_stats = {m: {"diff": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p": 1.0} for m in ["nDCG@10", "MRR@10", "Recall@100"]}
                    return {
                        "informal": informal,
                        "formal": formal,
                        "gap": gap,
                        "gap_stats": gap_stats
                    }

    # Fallback: try to load from npy files in llm_rewrite/cache
    llm_cache_dir = Path(cache_dir) / ".." / "llm_rewrite" / "cache"
    if not llm_cache_dir.exists():
        llm_cache_dir = HERE / "llm_rewrite" / "cache"

    if (llm_cache_dir / "formal_emb.npy").exists() and (llm_cache_dir / "informal_emb.npy").exists():
        # We have LLM normalization embeddings
        store = EmbeddingStore(cache_dir)
        with open(Path(cache_dir) / "splits.json") as f:
            splits = json.load(f)
        qids = splits[split]

        class LLMEmbeddingStore(EmbeddingStore):
            def __init__(self, cache_dir, device):
                super().__init__(cache_dir, device)
                # Override the informal embeddings with LLM-normalized ones
                self.informal = torch.from_numpy(np.load(str(llm_cache_dir / "informal_emb.npy"))).to(self.device)
                self.formal = torch.from_numpy(np.load(str(llm_cache_dir / "formal_emb.npy"))).to(self.device)

            def informal(self):
                return self.informal[self.rows].float()

            def formal(self):
                return self.formal[self.rows].float()

        llm_store = LLMEmbeddingStore(llm_cache_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        llm_evaluator = Evaluator(llm_store, qids, data_dir)

        # Evaluate baseline (no adaptation) on LLM-normalized queries
        return evaluate_baseline(llm_evaluator, is_tsdae=False, with_std=with_std)
    else:
        # Fallback to returning None to indicate we could not load the LLM normalization results
        return None


def evaluate_infonce_model(data_dir: str, split: str, with_std: bool = False) -> Dict:
    """Evaluate Query InfoNCE model."""
    infonce_cache_dir = HERE / "models" / "query_infonce" / "cache" / "infc_query_only"
    if not infonce_cache_dir.exists():
        return None

    store = EmbeddingStore(infonce_cache_dir)
    with open(infonce_cache_dir / "splits.json") as f:
        splits = json.load(f)
    qids = splits[split]
    evaluator = Evaluator(store, qids, data_dir)

    return evaluate_baseline(evaluator, is_tsdae=False, with_std=with_std)


MODEL_ORDER = [
    ("BGE-M3 Base",                  "baseline"),
    ("BGE-M3 + LLM Norm",            "llm_norm"),
    ("DANN",                         "dann"),  # Query DANN (methods/query_dann/outputs/v2_dann_g1.0, gamma=1.0)
    ("Query Distillation",  "dist_linear"),
    ("DANN + Adapter", "dann_adapter"),
    ("TSDAE",                        "tsdae"),
    ("TSDAE + Meaning",              "tsdae_meaning"),
    ("Query InfoNCE",                "infonce"),
]

METRICS_ALL = ["Recall@100", "MRR@10", "nDCG@10"]


def _get(d: dict, key: str, with_std: bool):
    """Return (mean, std) or (value, None) from a metric dict."""
    v = d[key]
    if with_std and isinstance(v, dict):
        return v["mean"], v["std"]
    return (v, None)


def _fmt(mean: float, std=None, width: int = 6) -> str:
    """Format a single metric value, optionally with ± std."""
    if std is not None:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def print_results_table(all_results: Dict, with_std: bool = False):
    """Two tables:
    1. Full table — Model | Formal (R@100, MRR@10, nDCG@10) | Informal (R@100, MRR@10, nDCG@10)
    2. nDCG-only table — Model | Formal nDCG@10 | Informal nDCG@10 | Gap
    """
    # ── helpers ──────────────────────────────────────────────────────────────
    col = 34  # model name column width
    mw  = 22 if with_std else 7  # per-metric cell width (wider for ±std)

    def header_cell(label, w):
        return f"{label:^{w}}"

    def row_missing(name):
        cells = " | ".join(["—" * mw] * 3)
        return f"{name:<{col}} | {cells} | {cells}"

    def metric_cells(side: dict, with_std: bool) -> str:
        parts = []
        for m in METRICS_ALL:
            mean, std = _get(side, m, with_std)
            parts.append(f"{_fmt(mean, std):^{mw}}")
        return " | ".join(parts)

    # ── Table 1: full metrics ─────────────────────────────────────────────────
    sep_inner = "-+-".join(["-" * mw] * 3)
    sep_outer = "-" * (col + 3 + len(sep_inner) + 3 + len(sep_inner))

    print("\n" + "=" * len(sep_outer))
    print("RESULTS TABLE — Formal vs Informal (R@100 | MRR@10 | nDCG@10)")
    print("=" * len(sep_outer))

    hdr_metrics = " | ".join(header_cell(m, mw) for m in METRICS_ALL)
    print(f"{'Model':<{col}} | {header_cell('Formal', len(hdr_metrics))} | {header_cell('Informal', len(hdr_metrics))}")
    print(f"{'':<{col}} | {hdr_metrics} | {hdr_metrics}")
    print(sep_outer)

    for display_name, key in MODEL_ORDER:
        if key not in all_results:
            print(row_missing(display_name))
            continue
        res = all_results[key]
        form_cells = metric_cells(res["formal"],   with_std)
        inf_cells  = metric_cells(res["informal"], with_std)
        print(f"{display_name:<{col}} | {form_cells} | {inf_cells}")

    print("=" * len(sep_outer))

    # ── Table 2: nDCG@10 only ────────────────────────────────────────────────
    ncol = 18 if with_std else 10   # cell width for nDCG columns
    gap_w = 15 if with_std else 8
    sep2 = "-" * (col + 3 + ncol + 3 + ncol + 3 + gap_w)

    print("\n" + "=" * len(sep2))
    print("nDCG@10 SUMMARY — Formal vs Informal")
    print("=" * len(sep2))
    print(f"{'Model':<{col}} | {'Formal nDCG@10':^{ncol}} | {'Informal nDCG@10':^{ncol}} | {'Gap (F-I)':^{gap_w}}")
    print(sep2)

    for display_name, key in MODEL_ORDER:
        if key not in all_results:
            print(f"{display_name:<{col}} | {'—':^{ncol}} | {'—':^{ncol}} | {'—':^{gap_w}}")
            continue
        res = all_results[key]
        fm, fs = _get(res["formal"],   "nDCG@10", with_std)
        im, is_ = _get(res["informal"], "nDCG@10", with_std)
        gm, gs  = _get(res["gap"],      "nDCG@10", with_std)
        f_str = f"{_fmt(fm, fs):^{ncol}}"
        i_str = f"{_fmt(im, is_):^{ncol}}"
        g_str = f"{'+' if gm >= 0 else ''}{_fmt(gm, gs):^{gap_w}}"
        print(f"{display_name:<{col}} | {f_str} | {i_str} | {g_str}")

    print("=" * len(sep2))


def print_gap_analysis(all_results: Dict, with_std: bool = False):
    """Gap-reduction table: how much each model narrows the formal–informal nDCG@10 gap."""
    print("\n" + "=" * 78)
    print("GAP REDUCTION ANALYSIS (nDCG@10, relative to BGE-M3 Base baseline)")
    print("=" * 78)

    if "baseline" not in all_results:
        print("Baseline not found.")
        return

    bm, bs = _get(all_results["baseline"]["gap"], "nDCG@10", with_std)
    print(f"\n  Baseline gap (Formal − Informal): {'+' if bm >= 0 else ''}{_fmt(bm, bs)}\n")
    print(f"  {'Model':<35} {'Gap':>14} {'Reduction':>14} {'Reduction %':>12}")
    print("  " + "-" * 74)

    for display_name, key in MODEL_ORDER:
        if key == "baseline" or key not in all_results:
            continue
        gm, gs = _get(all_results[key]["gap"], "nDCG@10", with_std)
        reduction = bm - gm
        pct = (reduction / abs(bm)) * 100 if bm != 0 else 0.0
        if with_std and gs is not None and bs is not None:
            red_std = float(np.sqrt(bs**2 + gs**2))
            gap_s   = f"{'+' if gm >= 0 else ''}{gm:.4f}±{gs:.4f}"
            red_s   = f"{reduction:+.4f}±{red_std:.4f}"
        else:
            gap_s = f"{'+' if gm >= 0 else ''}{gm:.4f}"
            red_s = f"{reduction:+.4f}"
        print(f"  {display_name:<35} {gap_s:>14} {red_s:>14} {pct:>+11.1f}%")

    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--with_std", action="store_true", help="Show standard deviation alongside mean metrics")
    ap.add_argument("--cache_dir", default=str(HERE / "methods" / "cache"))
    ap.add_argument("--data_dir", default=str(HERE / "data"))
    ap.add_argument("--llm_cache_dir", default=str(HERE / "llm_rewrite" / "cache"),
                    help="Cache directory for LLM normalization embeddings")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*80}")
    print(f"EVALUATING SELECTED MODELS ON {args.split.upper()} SPLIT")
    print(f"{'='*80}\n")

    # Load data
    base_store = EmbeddingStore(args.cache_dir, device=device)
    with open(Path(args.cache_dir) / "splits.json") as f:
        splits = json.load(f)

    base_evaluator = Evaluator(base_store, splits[args.split], args.data_dir)

    all_results = {}

    # 1. BGE-M3 Base (baseline)
    print("Evaluating: BGE-M3 Base...")
    all_results["baseline"] = evaluate_baseline(base_evaluator, is_tsdae=False, with_std=args.with_std)
    if args.with_std:
        print(f"  Informal nDCG@10: {all_results['baseline']['informal']['nDCG@10']['mean']:.4f} ±{all_results['baseline']['informal']['nDCG@10']['std']:.4f}")
        print(f"  Formal nDCG@10: {all_results['baseline']['formal']['nDCG@10']['mean']:.4f} ±{all_results['baseline']['formal']['nDCG@10']['std']:.4f}")
        print(f"  Gap: {all_results['baseline']['gap']['nDCG@10']['mean']:+.4f} ±{all_results['baseline']['gap']['nDCG@10']['std']:.4f}\n")
    else:
        print(f"  Informal nDCG@10: {all_results['baseline']['informal']['nDCG@10']:.4f}")
        print(f"  Formal nDCG@10: {all_results['baseline']['formal']['nDCG@10']:.4f}")
        print(f"  Gap: {all_results['baseline']['gap']['nDCG@10']:+.4f}\n")

    # 2. BGE-M3 + LLM Normalization
    print("Evaluating: BGE-M3 + LLM Normalization...")
    llm_result = load_llm_normalization_results(args.llm_cache_dir, args.data_dir, args.split, with_std=args.with_std)
    if llm_result is not None:
        all_results["llm_norm"] = llm_result
        if args.with_std:
            print(f"  Informal nDCG@10: {all_results['llm_norm']['informal']['nDCG@10']['mean']:.4f} ±{all_results['llm_norm']['informal']['nDCG@10']['std']:.4f}")
            print(f"  Formal nDCG@10: {all_results['llm_norm']['formal']['nDCG@10']['mean']:.4f} ±{all_results['llm_norm']['formal']['nDCG@10']['std']:.4f}")
            print(f"  Gap: {all_results['llm_norm']['gap']['nDCG@10']['mean']:+.4f} ±{all_results['llm_norm']['gap']['nDCG@10']['std']:.4f}\n")
        else:
            print(f"  Informal nDCG@10: {all_results['llm_norm']['informal']['nDCG@10']:.4f}")
            print(f"  Formal nDCG@10: {all_results['llm_norm']['formal']['nDCG@10']:.4f}")
            print(f"  Gap: {all_results['llm_norm']['gap']['nDCG@10']:+.4f}\n")
    else:
        print("  ⚠️  LLM normalization results not available, skipping\n")

    # 3. Query DANN (gamma=1.0, adversarial training)
    print("Evaluating: Query DANN...")
    dann_dir = Path(HERE / "methods" / "query_dann" / "outputs" / "v2_dann_g1.0")
    model = load_model(dann_dir, device)
    if model is not None:
        all_results["dann"] = evaluate_model(model, base_evaluator, with_std=args.with_std)
        if args.with_std:
            print(f"  Informal nDCG@10: {all_results['dann']['informal']['nDCG@10']['mean']:.4f} ±{all_results['dann']['informal']['nDCG@10']['std']:.4f}")
            print(f"  Formal nDCG@10: {all_results['dann']['formal']['nDCG@10']['mean']:.4f} ±{all_results['dann']['formal']['nDCG@10']['std']:.4f}")
            print(f"  Gap: {all_results['dann']['gap']['nDCG@10']['mean']:+.4f} ±{all_results['dann']['gap']['nDCG@10']['std']:.4f}\n")
        else:
            print(f"  Informal nDCG@10: {all_results['dann']['informal']['nDCG@10']:.4f}")
            print(f"  Formal nDCG@10: {all_results['dann']['formal']['nDCG@10']:.4f}")
            print(f"  Gap: {all_results['dann']['gap']['nDCG@10']:+.4f}\n")
    else:
        print("  ❌ Skipped (model not found)\n")

    # 4. Query Distillation (Linear)
    print("Evaluating: Query DANN (Linear)...")
    dann_linear_dir = Path(HERE / "methods" / "query_dann" / "outputs" / "query_dann_linear")
    model = load_model(dann_linear_dir, device)
    if model is not None:
        all_results["dann_linear"] = evaluate_model(model, base_evaluator, with_std=args.with_std)
        if args.with_std:
            print(f"  Informal nDCG@10: {all_results['dann_linear']['informal']['nDCG@10']['mean']:.4f} ±{all_results['dann_linear']['informal']['nDCG@10']['std']:.4f}")
            print(f"  Formal nDCG@10: {all_results['dann_linear']['formal']['nDCG@10']['mean']:.4f} ±{all_results['dann_linear']['formal']['nDCG@10']['std']:.4f}")
            print(f"  Gap: {all_results['dann_linear']['gap']['nDCG@10']['mean']:+.4f} ±{all_results['dann_linear']['gap']['nDCG@10']['std']:.4f}\n")
        else:
            print(f"  Informal nDCG@10: {all_results['dann_linear']['informal']['nDCG@10']:.4f}")
            print(f"  Formal nDCG@10: {all_results['dann_linear']['formal']['nDCG@10']:.4f}")
            print(f"  Gap: {all_results['dann_linear']['gap']['nDCG@10']:+.4f}\n")
    else:
        print("  ❌ Skipped (model not found)\n")

    # 4. Query Distillation (Linear)
    print("Evaluating: Query Distillation (Linear)...")
    dist_linear_dir = Path(HERE / "methods" / "query_distillation" / "outputs" / "query_dist_linear")
    model = load_model(dist_linear_dir, device)
    if model is not None:
        all_results["dist_linear"] = evaluate_model(model, base_evaluator, with_std=args.with_std)
        if args.with_std:
            print(f"  Informal nDCG@10: {all_results['dist_linear']['informal']['nDCG@10']['mean']:.4f} ±{all_results['dist_linear']['informal']['nDCG@10']['std']:.4f}")
            print(f"  Formal nDCG@10: {all_results['dist_linear']['formal']['nDCG@10']['mean']:.4f} ±{all_results['dist_linear']['formal']['nDCG@10']['std']:.4f}")
            print(f"  Gap: {all_results['dist_linear']['gap']['nDCG@10']['mean']:+.4f} ±{all_results['dist_linear']['gap']['nDCG@10']['std']:.4f}\n")
        else:
            print(f"  Informal nDCG@10: {all_results['dist_linear']['informal']['nDCG@10']:.4f}")
            print(f"  Formal nDCG@10: {all_results['dist_linear']['formal']['nDCG@10']:.4f}")
            print(f"  Gap: {all_results['dist_linear']['gap']['nDCG@10']:+.4f}\n")
    else:
        print("  ❌ Skipped (model not found)\n")

    # 5. Query DANN + Adapter (Zero Init)
    print("Evaluating: Query DANN + Adapter (Zero Init)...")
    dann_adapter_dir = Path(HERE / "methods" / "query_dann" / "outputs" / "query_dann_adapter")
    model = load_model(dann_adapter_dir, device)
    if model is not None:
        all_results["dann_adapter"] = evaluate_model(model, base_evaluator, with_std=args.with_std)
        if args.with_std:
            print(f"  Informal nDCG@10: {all_results['dann_adapter']['informal']['nDCG@10']['mean']:.4f} ±{all_results['dann_adapter']['informal']['nDCG@10']['std']:.4f}")
            print(f"  Formal nDCG@10: {all_results['dann_adapter']['formal']['nDCG@10']['mean']:.4f} ±{all_results['dann_adapter']['formal']['nDCG@10']['std']:.4f}")
            print(f"  Gap: {all_results['dann_adapter']['gap']['nDCG@10']['mean']:+.4f} ±{all_results['dann_adapter']['gap']['nDCG@10']['std']:.4f}\n")
        else:
            print(f"  Informal nDCG@10: {all_results['dann_adapter']['informal']['nDCG@10']:.4f}")
            print(f"  Formal nDCG@10: {all_results['dann_adapter']['formal']['nDCG@10']:.4f}")
            print(f"  Gap: {all_results['dann_adapter']['gap']['nDCG@10']:+.4f}\n")
    else:
        print("  ❌ Skipped (model not found)\n")

    # 6. TSDAE
    print("Evaluating: TSDAE...")
    # Check if TSDAE cache exists
    tsdae_cache_dir = Path(HERE / "models" / "tsdae_sapt" / "cache" / "sapt_symmetric")
    tsdae_available = (tsdae_cache_dir / "formal_emb.npy").exists() \
        and (tsdae_cache_dir / "corpus_emb.npy").exists() \
        and (tsdae_cache_dir / "hard_negs.json").exists()

    if tsdae_available:
        tsdae_store = EmbeddingStore(tsdae_cache_dir, device=device)
        tsdae_evaluator = Evaluator(tsdae_store, splits[args.split], args.data_dir)
        all_results["tsdae"] = evaluate_baseline(tsdae_evaluator, is_tsdae=True, with_std=args.with_std)
        if args.with_std:
            print(f"  Informal nDCG@10: {all_results['tsdae']['informal']['nDCG@10']['mean']:.4f} ±{all_results['tsdae']['informal']['nDCG@10']['std']:.4f}")
            print(f"  Formal nDCG@10: {all_results['tsdae']['formal']['nDCG@10']['mean']:.4f} ±{all_results['tsdae']['formal']['nDCG@10']['std']:.4f}")
            print(f"  Gap: {all_results['tsdae']['gap']['nDCG@10']['mean']:+.4f} ±{all_results['tsdae']['gap']['nDCG@10']['std']:.4f}\n")
        else:
            print(f"  Informal nDCG@10: {all_results['tsdae']['informal']['nDCG@10']:.4f}")
            print(f"  Formal nDCG@10: {all_results['tsdae']['formal']['nDCG@10']:.4f}")
            print(f"  Gap: {all_results['tsdae']['gap']['nDCG@10']:+.4f}\n")
    else:
        print("  ⚠️  TSDAE cache not found, skipping\n")
        print(f"     Run: python3 models/tsdae_sapt/prepare_sapt_cache.py --mode symmetric "
              f"--out_cache {tsdae_cache_dir}\n")

    # 7. TSDAE + Meaning
    print("Evaluating: TSDAE + Meaning...")
    # Check if TSDAE + Meaning cache exists
    tsdae_meaning_cache_dir = Path(HERE / "tsdae_meaning" / "tsdae_meaning" / "cache" / "tsdae_meaning")
    tsdae_meaning_available = (tsdae_meaning_cache_dir / "formal_emb.npy").exists() \
        and (tsdae_meaning_cache_dir / "corpus_emb.npy").exists() \
        and (tsdae_meaning_cache_dir / "hard_negs.json").exists()

    if tsdae_meaning_available:
        tsdae_meaning_store = EmbeddingStore(tsdae_meaning_cache_dir, device=device)
        tsdae_meaning_evaluator = Evaluator(tsdae_meaning_store, splits[args.split], args.data_dir)
        all_results["tsdae_meaning"] = evaluate_baseline(tsdae_meaning_evaluator, is_tsdae=True, with_std=args.with_std)
        if args.with_std:
            print(f"  Informal nDCG@10: {all_results['tsdae_meaning']['informal']['nDCG@10']['mean']:.4f} ±{all_results['tsdae_meaning']['informal']['nDCG@10']['std']:.4f}")
            print(f"  Formal nDCG@10: {all_results['tsdae_meaning']['formal']['nDCG@10']['mean']:.4f} ±{all_results['tsdae_meaning']['formal']['nDCG@10']['std']:.4f}")
            print(f"  Gap: {all_results['tsdae_meaning']['gap']['nDCG@10']['mean']:+.4f} ±{all_results['tsdae_meaning']['gap']['nDCG@10']['std']:.4f}\n")
        else:
            print(f"  Informal nDCG@10: {all_results['tsdae_meaning']['informal']['nDCG@10']:.4f}")
            print(f"  Formal nDCG@10: {all_results['tsdae_meaning']['formal']['nDCG@10']:.4f}")
            print(f"  Gap: {all_results['tsdae_meaning']['gap']['nDCG@10']:+.4f}\n")
    else:
        print("  ⚠️  TSDAE + Meaning cache not found, skipping\n")
        print(f"     Run: python3 tsdae_meaning/tsdae_meaning/prepare_cache.py "
              f"--out_cache {tsdae_meaning_cache_dir}\n")

    # 7. Query InfoNCE
    print("Evaluating: Query InfoNCE...")
    infonce_result = evaluate_infonce_model(args.data_dir, args.split, with_std=args.with_std)
    if infonce_result is not None:
        all_results["infonce"] = infonce_result
        if args.with_std:
            print(f"  Informal nDCG@10: {all_results['infonce']['informal']['nDCG@10']['mean']:.4f} ±{all_results['infonce']['informal']['nDCG@10']['std']:.4f}")
            print(f"  Formal nDCG@10: {all_results['infonce']['formal']['nDCG@10']['mean']:.4f} ±{all_results['infonce']['formal']['nDCG@10']['std']:.4f}")
            print(f"  Gap: {all_results['infonce']['gap']['nDCG@10']['mean']:+.4f} ±{all_results['infonce']['gap']['nDCG@10']['std']:.4f}\n")
        else:
            print(f"  Informal nDCG@10: {all_results['infonce']['informal']['nDCG@10']:.4f}")
            print(f"  Formal nDCG@10: {all_results['infonce']['formal']['nDCG@10']:.4f}")
            print(f"  Gap: {all_results['infonce']['gap']['nDCG@10']:+.4f}\n")
    else:
        print("  ⚠️  Query InfoNCE model not found or not implemented, skipping\n")

    # Print results
    print_results_table(all_results, with_std=args.with_std)
    print_gap_analysis(all_results, with_std=args.with_std)

    # Save results
    results_json = Path(HERE / "selected_models_results.json")
    with open(results_json, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        json.dump(convert_for_json(all_results), f, indent=2)

    print(f"\n✅ Results saved to: {results_json}\n")
    print(f"{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
