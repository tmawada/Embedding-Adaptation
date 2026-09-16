"""Test-set evaluation of every method checkpoint for one encoder, with paired bootstrap tests.

  python3 encoder_comparison/evaluate_methods.py --encoder me5_large_instruct
  python3 encoder_comparison/evaluate_methods.py --encoder bge_m3     # re-evaluates the existing BGE-M3 runs

Same protocol as ../query_distillation/evaluate.py (reuses its adapter loader), with encoder-aware row labels.
Families: query_dann_v2 (best DANN gamma picked on dev, plus gamma=0), query_distillation (4 variants),
uda_disjoint and uda_overlap (strict best_source checkpoints). Writes runs/<encoder>/test_<family>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "query_distillation"))

from dataset import EmbeddingStore  # noqa: E402
from eval import METRICS, Evaluator, paired_bootstrap  # noqa: E402
from evaluate import adapt, load_adapter  # noqa: E402  (query_distillation/evaluate.py)
from encoders import ENCODERS  # noqa: E402

FAMILIES = ("query_dann_v2", "query_distillation", "uda_disjoint", "uda_overlap")


def best_by_dev(ckpt_dirs):
    return max(ckpt_dirs, key=lambda d: json.load(open(os.path.join(d, "config.json")))["dev"]["nDCG@10"])


def checkpoints(encoder: str) -> dict:
    """{family: ({run name: checkpoint dir}, reference run name or None)}."""
    if encoder == "bge_m3":
        dann = glob.glob(os.path.join(ROOT, "runs", "v2_dann_g*", "best"))
        gamma0 = os.path.join(ROOT, "runs", "v2_infonce_only", "best")
        dist = os.path.join(ROOT, "query_distillation", "runs")
        uda = os.path.join(ROOT, "uda_query_dann", "runs")
    else:
        base = os.path.join(HERE, "runs", encoder)
        dann = [d for d in glob.glob(os.path.join(base, "query_dann_v2", "g*", "best"))
                if os.path.basename(os.path.dirname(d)) != "g0"]
        gamma0 = os.path.join(base, "query_dann_v2", "g0", "best")
        dist = os.path.join(base, "query_distillation")
        uda = os.path.join(base, "uda_query_dann")
    if not dann:
        raise SystemExit(f"no Query-DANN v2 checkpoints found for {encoder}")
    return {
        "query_dann_v2": ({"query_dann_v2": best_by_dev(dann), "infonce_only": gamma0}, "infonce_only"),
        "query_distillation": ({n: os.path.join(dist, n, "best") for n in ("embed", "score", "embed_infonce", "score_infonce")}, None),
        "uda_disjoint": ({"source_only": os.path.join(uda, "disjoint_g0", "best_source"),
                          **{f"uda_g{g}": os.path.join(uda, f"disjoint_g{g}", "best_source") for g in ("0.1", "0.5", "1.0")}},
                         "source_only"),
        "uda_overlap": ({"source_only": os.path.join(uda, "overlap_g0", "best_source"),
                         "uda_g1.0": os.path.join(uda, "overlap_g1.0", "best_source")}, "source_only"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENCODERS))
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()
    spec = ENCODERS[args.encoder]
    out_dir = os.path.join(HERE, "runs", args.encoder)
    os.makedirs(out_dir, exist_ok=True)

    ckpts = checkpoints(args.encoder)
    missing = [p for fam in args.families for p in ckpts[fam][0].values() if not os.path.exists(os.path.join(p, "adapter.pt"))]
    if missing:
        raise SystemExit("missing checkpoints:\n  " + "\n  ".join(missing))

    store = EmbeddingStore(spec.cache_dir)
    with open(os.path.join(spec.cache_dir, "splits.json")) as f:
        ev = Evaluator(store, json.load(f)[args.split], args.data_dir)
    formal_row = f"base {spec.label} | formal (upper bound)"
    zero_row = f"base {spec.label} | informal (zero-shot)"
    base = {formal_row: ev.run_per_query(ev.formal()), zero_row: ev.run_per_query(ev.informal())}

    for family in args.families:
        runs, reference = ckpts[family]
        per_query = dict(base)
        for name, path in runs.items():
            adapter = load_adapter(path, store.device)
            per_query[f"{name} | informal"] = ev.run_per_query(adapt(adapter, ev.informal()))
            per_query[f"{name} | formal"] = ev.run_per_query(adapt(adapter, ev.formal()))
        results = {row: {m: float(v.mean()) for m, v in pq.items()} for row, pq in per_query.items()}

        upper, lower = results[formal_row]["nDCG@10"], results[zero_row]["nDCG@10"]
        width = max(len(r) for r in results) + 2
        print(f"\n[{spec.label} | {family}] MIRACL-id {args.split}: {len(ev.qids)} queries")
        print(f"{'condition':<{width}} {'nDCG@10':>8} {'MRR@10':>8} {'R@100':>8} {'gap closed':>11}")
        for row, m in results.items():
            gap = f"{100 * (m['nDCG@10'] - lower) / (upper - lower):>10.1f}%" if row.endswith("| informal") else ""
            print(f"{row:<{width}} {m['nDCG@10']:>8.4f} {m['MRR@10']:>8.4f} {m['Recall@100']:>8.4f} {gap:>11}")

        comparisons = []
        for name in runs:
            comparisons.append((f"{name} vs zero-shot", f"{name} | informal", zero_row))
            if reference and name != reference:
                comparisons.append((f"{name} vs {reference}", f"{name} | informal", f"{reference} | informal"))
            comparisons.append((f"{name} formal vs base formal", f"{name} | formal", formal_row))
        significance = {}
        cw = max(len(c[0]) for c in comparisons) + 2
        print(f"Paired bootstrap ({args.bootstrap} samples; * = p < 0.05)")
        for label, a, b in comparisons:
            significance[label] = {}
            for m in METRICS:
                r = paired_bootstrap(per_query[a][m], per_query[b][m], args.bootstrap)
                significance[label][m] = r
                ci = f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
                print(f"  {label:<{cw}} {m:<11} {r['diff']:>+8.4f} {ci:>20} {r['p']:>7.3f}{' *' if r['p'] < 0.05 else ''}")

        out = os.path.join(out_dir, f"test_{family}.json")
        with open(out, "w") as f:
            json.dump({"encoder": args.encoder, "label": spec.label, "family": family, "split": args.split,
                       "n_queries": len(ev.qids), "reference": reference,
                       "runs": {n: os.path.relpath(p, ROOT) for n, p in runs.items()},
                       "base_formal": results[formal_row], "base_informal": results[zero_row],
                       "results": results, "bootstrap_samples": args.bootstrap, "significance": significance}, f, indent=2)
        print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
