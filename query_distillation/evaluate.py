"""Test-set comparison of adapter checkpoints with paired bootstrap significance tests.

  python3 evaluate.py --run source_only=runs/disjoint_g0/best_source --run uda_g1.0=runs/disjoint_g1.0/best_source \
      --reference source_only --out runs/test_strict.json

Rows: base BGE-M3 on formal (upper bound) and informal (zero-shot) queries, then every --run on informal queries
(with % of the formal/informal gap closed) and on formal queries (robustness). Bootstrap comparisons: each run vs
zero-shot, each run vs --reference (if given), and each run's formal queries vs base formal.
Any checkpoint directory with adapter.pt + config.json works, including Query-DANN v2 (../runs/v2_dann_g1.0/best).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset import EmbeddingStore  # noqa: E402
from eval import METRICS, Evaluator, paired_bootstrap  # noqa: E402
from model import ADAPTERS, EMB_DIM  # noqa: E402

FORMAL = "base BGE-M3 | formal (upper bound)"
ZERO_SHOT = "base BGE-M3 | informal (zero-shot)"


def load_adapter(ckpt_dir: str, device) -> torch.nn.Module:
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    adapter = ADAPTERS[cfg.get("adapter", "layernorm")](EMB_DIM, cfg["bottleneck"], cfg["adapter_dropout"]).to(device)
    adapter.load_state_dict(torch.load(os.path.join(ckpt_dir, "adapter.pt"), map_location=device))
    return adapter.eval()


@torch.no_grad()
def adapt(adapter: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    return F.normalize(adapter(z).float(), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=checkpoint_dir (repeatable)")
    ap.add_argument("--reference", default=None, help="Run name every other run is also compared against")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    store = EmbeddingStore(args.cache_dir)
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        ev = Evaluator(store, json.load(f)[args.split], args.data_dir)

    per_query = {FORMAL: ev.run_per_query(ev.formal()), ZERO_SHOT: ev.run_per_query(ev.informal())}
    names = []
    for spec in args.run:
        name, path = spec.split("=", 1)
        adapter = load_adapter(path, store.device)
        per_query[f"{name} | informal"] = ev.run_per_query(adapt(adapter, ev.informal()))
        per_query[f"{name} | formal"] = ev.run_per_query(adapt(adapter, ev.formal()))
        names.append(name)
    if args.reference and args.reference not in names:
        raise SystemExit(f"--reference {args.reference} is not one of the --run names {names}")
    results = {row: {m: float(v.mean()) for m, v in pq.items()} for row, pq in per_query.items()}

    upper, lower = results[FORMAL]["nDCG@10"], results[ZERO_SHOT]["nDCG@10"]
    width = max(len(r) for r in results) + 2
    print(f"\nMIRACL-id {args.split} split: {len(ev.qids)} queries, {len(store.doc_ids):,} passages")
    print(f"{'condition':<{width}} {'nDCG@10':>8} {'MRR@10':>8} {'R@100':>8} {'gap closed':>11}")
    for row, m in results.items():
        gap = f"{100 * (m['nDCG@10'] - lower) / (upper - lower):>10.1f}%" if row.endswith("| informal") else ""
        print(f"{row:<{width}} {m['nDCG@10']:>8.4f} {m['MRR@10']:>8.4f} {m['Recall@100']:>8.4f} {gap:>11}")

    comparisons = []
    for name in names:
        comparisons.append((f"{name} vs zero-shot", f"{name} | informal", ZERO_SHOT))
        if args.reference and name != args.reference:
            comparisons.append((f"{name} vs {args.reference}", f"{name} | informal", f"{args.reference} | informal"))
        comparisons.append((f"{name} formal vs base formal", f"{name} | formal", FORMAL))

    significance = {}
    if args.bootstrap > 0:
        cw = max(len(c[0]) for c in comparisons) + 2
        print(f"\nPaired bootstrap ({args.bootstrap} samples, two-sided; * = p < 0.05)")
        print(f"{'comparison':<{cw}} {'metric':<11} {'diff':>8} {'95% CI':>20} {'p':>7}")
        for label, a, b in comparisons:
            significance[label] = {}
            for m in METRICS:
                r = paired_bootstrap(per_query[a][m], per_query[b][m], args.bootstrap)
                significance[label][m] = r
                ci = f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
                print(f"{label:<{cw}} {m:<11} {r['diff']:>+8.4f} {ci:>20} {r['p']:>7.3f}{' *' if r['p'] < 0.05 else ''}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "n_queries": len(ev.qids), "runs": dict(s.split("=", 1) for s in args.run),
                       "results": results, "bootstrap_samples": args.bootstrap, "significance": significance}, f, indent=2)


if __name__ == "__main__":
    main()
