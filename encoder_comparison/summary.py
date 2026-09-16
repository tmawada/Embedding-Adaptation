"""Collate the encoder comparison into one table: runs/summary.md + runs/summary.json.

Needs runs/zero_shot.json (zero_shot.py) and runs/<encoder>/test_<family>.json (evaluate_methods.py).
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

ROWS = [  # (row label, family, run name)
    ("Query-DANN v2 (γ picked on dev)", "query_dann_v2", "query_dann_v2"),
    ("Adapter + InfoNCE only (γ=0)", "query_dann_v2", "infonce_only"),
    ("Distillation: embed", "query_distillation", "embed"),
    ("Distillation: score", "query_distillation", "score"),
    ("Distillation: embed + InfoNCE", "query_distillation", "embed_infonce"),
    ("Distillation: score + InfoNCE", "query_distillation", "score_infonce"),
    ("UDA disjoint: source-only (γ=0)", "uda_disjoint", "source_only"),
    ("UDA disjoint: γ=0.1", "uda_disjoint", "uda_g0.1"),
    ("UDA disjoint: γ=0.5", "uda_disjoint", "uda_g0.5"),
    ("UDA disjoint: γ=1.0", "uda_disjoint", "uda_g1.0"),
    ("UDA overlap: source-only (γ=0)", "uda_overlap", "source_only"),
    ("UDA overlap: γ=1.0", "uda_overlap", "uda_g1.0"),
]


def cell(res: dict, run: str) -> dict:
    inf, form = res["results"][f"{run} | informal"], res["results"][f"{run} | formal"]
    lo, up = res["base_informal"]["nDCG@10"], res["base_formal"]["nDCG@10"]
    s = res["significance"][f"{run} vs zero-shot"]["nDCG@10"]
    sf = res["significance"][f"{run} formal vs base formal"]["nDCG@10"]
    return {"nDCG@10": inf["nDCG@10"], "MRR@10": inf["MRR@10"], "Recall@100": inf["Recall@100"],
            "gap_closed_pct": 100 * (inf["nDCG@10"] - lo) / (up - lo),
            "delta_vs_zero_shot": s["diff"], "p_vs_zero_shot": s["p"],
            "formal_delta": sf["diff"], "p_formal_delta": sf["p"], "checkpoint": res["runs"][run]}


def pts(diff: float, p: float) -> str:
    return f"{100 * diff:+.2f}{'*' if p < 0.05 else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["bge_m3", "me5_large_instruct"])
    args = ap.parse_args()

    with open(os.path.join(RUNS, "zero_shot.json")) as f:
        zs = json.load(f)
    tests = {e: {} for e in args.encoders}
    for e in args.encoders:
        for fam in {r[1] for r in ROWS}:
            with open(os.path.join(RUNS, e, f"test_{fam}.json")) as f:
                tests[e][fam] = json.load(f)
    labels = {e: zs["encoders"][e]["label"] for e in args.encoders}
    methods = {row: {e: cell(tests[e][fam], run) for e in args.encoders} for row, fam, run in ROWS}

    L = [f"# Embedding model comparison — MIRACL-id {zs['split']} split",
         "", f"{zs['n_queries']} held-out queries, 500,000 passages. Δ values are nDCG@10 points; "
         f"* = p < 0.05 (paired bootstrap, {zs['bootstrap_samples']} samples). Identical splits, methods, "
         "hyperparameters and step budgets for both encoders.", "", "## Zero-shot robustness", "",
         "| Encoder | Formal nDCG@10 | Informal nDCG@10 | Drop (pts) | Drop % | Formal MRR@10 | Informal MRR@10 | Formal R@100 | Informal R@100 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for e in args.encoders:
        r = zs["encoders"][e]
        sig = zs["significance"][f"{r['label']}: informal - formal"]["nDCG@10"]
        L.append(f"| {r['label']} | {r['formal']['nDCG@10']:.4f} | {r['informal']['nDCG@10']:.4f} | "
                 f"{pts(r['drop']['nDCG@10'], sig['p'])} | {r['drop_pct']['nDCG@10']:+.1f}% | {r['formal']['MRR@10']:.4f} | "
                 f"{r['informal']['MRR@10']:.4f} | {r['formal']['Recall@100']:.4f} | {r['informal']['Recall@100']:.4f} |")
    for key, s in zs["significance"].items():
        if " - " in key and "(" in key:
            m = s["nDCG@10"]
            L.append(f"\n- {key}: {pts(m['diff'], m['p'])} nDCG@10 (p={m['p']:.3f})")
    L += ["", "Cosine-score scale (formal queries): " + "; ".join(
        f"{zs['encoders'][e]['label']} positive {zs['encoders'][e]['score_stats']['positive']:.3f} / "
        f"top-10 non-relevant {zs['encoders'][e]['score_stats']['top10_non_relevant']:.3f} / "
        f"random {zs['encoders'][e]['score_stats']['random_passage']:.3f}" for e in args.encoders)]

    head = " | ".join(f"{labels[e]} nDCG@10 | gap closed | Δ vs zero-shot | formal Δ" for e in args.encoders)
    L += ["", "## Methods on informal queries: nDCG@10", "", f"| Method | {head} |", "|---|" + "---|" * (4 * len(args.encoders))]
    for row, cells in methods.items():
        L.append(f"| {row} | " + " | ".join(
            f"{c['nDCG@10']:.4f} | {c['gap_closed_pct']:.1f}% | {pts(c['delta_vs_zero_shot'], c['p_vs_zero_shot'])} | "
            f"{pts(c['formal_delta'], c['p_formal_delta'])}" for c in (cells[e] for e in args.encoders)) + " |")

    head = " | ".join(f"{labels[e]} MRR@10 | {labels[e]} R@100" for e in args.encoders)
    L += ["", "## Methods on informal queries: MRR@10 and Recall@100", "", f"| Method | {head} |", "|---|" + "---|" * (2 * len(args.encoders))]
    for row, cells in methods.items():
        L.append(f"| {row} | " + " | ".join(f"{cells[e]['MRR@10']:.4f} | {cells[e]['Recall@100']:.4f}" for e in args.encoders) + " |")

    L += ["", "## Selected Query-DANN v2 checkpoints (γ picked on dev nDCG@10)", ""]
    L += [f"- {labels[e]}: `{methods['Query-DANN v2 (γ picked on dev)'][e]['checkpoint']}`" for e in args.encoders]

    text = "\n".join(L) + "\n"
    with open(os.path.join(RUNS, "summary.md"), "w") as f:
        f.write(text)
    with open(os.path.join(RUNS, "summary.json"), "w") as f:
        json.dump({"zero_shot": zs, "methods": methods}, f, indent=2, ensure_ascii=False)
    print(text)


if __name__ == "__main__":
    main()
