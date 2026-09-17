"""Paired-bootstrap head-to-heads for the stacked model (formal and informal nDCG@10, test split).

  python3 stacked/head_to_head.py --stack stacked/runs/sapt_queries/kd1.0/best --base_kd stacked/runs/base/kd1.0/best
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "query_distillation"))

from dataset import EmbeddingStore  # noqa: E402
from eval import Evaluator, paired_bootstrap  # noqa: E402
from evaluate import adapt, load_adapter  # noqa: E402

H = os.path.join(ROOT, "comparison", "runs", "harmonized")
A = os.path.join(H, "ablation_adapter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, help="TSDAE + DANN + KD checkpoint")
    ap.add_argument("--base_kd", required=True, help="DANN + KD checkpoint on the base tower")
    ap.add_argument("--tsdae_cache", default=os.path.join(H, "sapt_queries", "cache"))
    ap.add_argument("--dann_linear", default=os.path.join(A, "dann", "linear", "best"))
    ap.add_argument("--distill_linear", default=os.path.join(A, "distill", "linear", "best"))
    ap.add_argument("--tsdae_dann_linear", default=os.path.join(A, "sapt_queries", "dann_linear", "best"))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "head_to_head.json"))
    args = ap.parse_args()

    store = EmbeddingStore(os.path.join(ROOT, "cache"))
    with open(os.path.join(ROOT, "cache", "splits.json")) as f:
        ev = Evaluator(store, json.load(f)["test"], os.path.join(ROOT, "data"))
    base = {"formal": ev.formal(), "informal": ev.informal()}
    with open(os.path.join(args.tsdae_cache, "query_ids.json")) as f:
        order = {q: i for i, q in enumerate(json.load(f))}
    sel = [order[q] for q in ev.qids]
    tsdae = {k: F.normalize(torch.from_numpy(np.asarray(np.load(os.path.join(args.tsdae_cache, f"{k}_emb.npy"),
                                                                 mmap_mode="r")[sel], dtype=np.float32)).to(store.device), dim=-1)
             for k in ("formal", "informal")}

    def run(ckpt, tower):
        a = load_adapter(ckpt, store.device)
        return {k: ev.run_per_query(adapt(a, v))["nDCG@10"] for k, v in tower.items()}

    systems = {
        "TSDAE + DANN + KD (stack)": run(args.stack, tsdae),
        "DANN + KD (base tower)": run(args.base_kd, base),
        "DANN (linear)": run(args.dann_linear, base),
        "TSDAE + DANN (linear)": run(args.tsdae_dann_linear, tsdae),
        "Distillation score (linear)": run(args.distill_linear, base),
        "BGE-M3 base": {k: ev.run_per_query(v)["nDCG@10"] for k, v in base.items()},
    }
    pairs = [("TSDAE + DANN + KD (stack)", b) for b in
             ("DANN (linear)", "TSDAE + DANN (linear)", "Distillation score (linear)", "DANN + KD (base tower)", "BGE-M3 base")]
    pairs.append(("DANN + KD (base tower)", "DANN (linear)"))

    out = {}
    print(f"\nnDCG@10 head-to-head, test ({len(ev.qids)} queries), paired bootstrap {args.bootstrap}")
    print(f"{'A':<28} {'B':<28} {'side':<9} {'A':>7} {'B':>7} {'A-B':>7} {'95% CI':>18} {'p':>6}")
    for a, b in pairs:
        for side in ("informal", "formal"):
            r = paired_bootstrap(systems[a][side], systems[b][side], args.bootstrap)
            out[f"{a} vs {b} ({side})"] = r
            ci = f"[{100 * r['ci95_low']:+.2f}, {100 * r['ci95_high']:+.2f}]"
            print(f"{a:<28} {b:<28} {side:<9} {systems[a][side].mean():>7.4f} {systems[b][side].mean():>7.4f} "
                  f"{100 * r['diff']:>+7.2f} {ci:>18} {r['p']:>6.3f}{' *' if r['p'] < 0.05 else ''}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"stack": args.stack, "base_kd": args.base_kd, "n_queries": len(ev.qids),
                   "means": {n: {k: float(v.mean()) for k, v in m.items()} for n, m in systems.items()},
                   "tests": out}, f, indent=2)


if __name__ == "__main__":
    main()
