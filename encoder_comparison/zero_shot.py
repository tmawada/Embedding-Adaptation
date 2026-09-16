"""Zero-shot robustness of each encoder to informal queries (formal vs informal), with paired bootstrap tests.

  python3 encoder_comparison/zero_shot.py        # BGE-M3 and mE5-large-instruct on the test split

Also reports each encoder's cosine-score scale (labelled positives, top-10 non-relevant, random passages),
which matters because the adaptation methods use fixed softmax temperatures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset import EmbeddingStore  # noqa: E402
from eval import METRICS, Evaluator, dense_search, paired_bootstrap  # noqa: E402
from encoders import ENCODERS  # noqa: E402


@torch.no_grad()
def score_stats(store: EmbeddingStore, ev: Evaluator) -> dict:
    """Mean cosine of formal queries with: labelled positives, top-10 non-relevant results, 1000 random passages."""
    z = ev.formal()
    scores, idx = dense_search(z, store.corpus, 100)
    pos, hard = [], []
    for i, q in enumerate(ev.qids):
        rel = torch.tensor([store.doc2row[d] for d in ev.qrels[q]], device=store.device)
        pos.append((store.corpus[rel].float() @ z[i]).mean().item())
        non_rel = [s for s, j in zip(scores[i].tolist(), idx[i].tolist()) if store.doc_ids[j] not in ev.qrels[q]][:10]
        hard.append(float(np.mean(non_rel)))
    rows = torch.randint(0, len(store.doc_ids), (1000,), generator=torch.Generator().manual_seed(0)).to(store.device)
    return {"positive": float(np.mean(pos)), "top10_non_relevant": float(np.mean(hard)),
            "random_passage": (z @ store.corpus[rows].float().T).mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["bge_m3", "me5_large_instruct"], choices=list(ENCODERS))
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "zero_shot.json"))
    args = ap.parse_args()

    per_query, report, ref_qids = {}, {}, None
    for name in args.encoders:
        spec = ENCODERS[name]
        store = EmbeddingStore(spec.cache_dir)
        with open(os.path.join(spec.cache_dir, "splits.json")) as f:
            ev = Evaluator(store, json.load(f)[args.split], args.data_dir)
        ref_qids = ref_qids or ev.qids
        assert ev.qids == ref_qids, f"{name}: evaluated queries differ from {args.encoders[0]}"
        pq = {"formal": ev.run_per_query(ev.formal()), "informal": ev.run_per_query(ev.informal())}
        per_query[name] = pq
        means = {c: {m: float(pq[c][m].mean()) for m in METRICS} for c in pq}
        report[name] = {"label": spec.label, "model_id": spec.model_id, **means,
                        "drop": {m: means["informal"][m] - means["formal"][m] for m in METRICS},
                        "drop_pct": {m: 100 * (means["informal"][m] - means["formal"][m]) / means["formal"][m] for m in METRICS},
                        "score_stats": score_stats(store, ev)}
        del store, ev
        torch.cuda.empty_cache()

    print(f"\nZero-shot, MIRACL-id {args.split} split: {len(ref_qids)} queries, 500,000 passages")
    print(f"{'encoder':<20} | {'formal nDCG@10':>14} {'MRR@10':>7} {'R@100':>7} | {'informal nDCG@10':>16} {'MRR@10':>7} {'R@100':>7} | {'nDCG@10 drop':>16}")
    for name, r in report.items():
        f, i = r["formal"], r["informal"]
        print(f"{r['label']:<20} | {f['nDCG@10']:>14.4f} {f['MRR@10']:>7.4f} {f['Recall@100']:>7.4f} | "
              f"{i['nDCG@10']:>16.4f} {i['MRR@10']:>7.4f} {i['Recall@100']:>7.4f} | "
              f"{100 * r['drop']['nDCG@10']:>+7.2f} pts ({r['drop_pct']['nDCG@10']:+.1f}%)")
    print("\ncosine-score scale (formal queries):  positive | top-10 non-relevant | random passage")
    for name, r in report.items():
        s = r["score_stats"]
        print(f"  {r['label']:<20} {s['positive']:.4f} | {s['top10_non_relevant']:.4f} | {s['random_passage']:.4f}")

    comparisons = [(f"{report[n]['label']}: informal - formal", per_query[n]["informal"], per_query[n]["formal"])
                   for n in args.encoders]
    first = args.encoders[0]
    for name in args.encoders[1:]:
        a, b = per_query[name], per_query[first]
        la, lb = report[name]["label"], report[first]["label"]
        comparisons.append((f"{la} - {lb} (formal)", a["formal"], b["formal"]))
        comparisons.append((f"{la} - {lb} (informal)", a["informal"], b["informal"]))
        drop_a = {m: a["informal"][m] - a["formal"][m] for m in METRICS}
        drop_b = {m: b["informal"][m] - b["formal"][m] for m in METRICS}
        comparisons.append((f"{la} - {lb} (informal drop)", drop_a, drop_b))

    significance = {}
    cw = max(len(c[0]) for c in comparisons) + 2
    print(f"\nPaired bootstrap ({args.bootstrap} samples, two-sided; * = p < 0.05)")
    print(f"{'comparison':<{cw}} {'metric':<11} {'diff':>8} {'95% CI':>20} {'p':>7}")
    for label, x, y in comparisons:
        significance[label] = {}
        for m in METRICS:
            r = paired_bootstrap(x[m], y[m], args.bootstrap)
            significance[label][m] = r
            ci = f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
            print(f"{label:<{cw}} {m:<11} {r['diff']:>+8.4f} {ci:>20} {r['p']:>7.3f}{' *' if r['p'] < 0.05 else ''}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"split": args.split, "n_queries": len(ref_qids), "encoders": report,
                   "bootstrap_samples": args.bootstrap, "significance": significance}, f, indent=2)


if __name__ == "__main__":
    main()
