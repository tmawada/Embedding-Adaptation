"""Zero-shot evaluation of the SAPT (TSDAE) model against base BGE-M3 — no adapter involved.

Conditions (same test queries, same 500k passages):
  base              original BGE-M3 for queries and passages            (../cache)
  sapt_query_only   SAPT queries vs original passage embeddings          (cache/sapt_query_only)
  sapt_symmetric    SAPT for queries and passages                        (cache/sapt_symmetric)
Reports formal / informal nDCG@10, MRR@10, Recall@100, the informal drop, paired bootstrap vs base, and
embedding diagnostics: cos(informal, formal twin) and how far SAPT moved queries / passages from BGE-M3.
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
from eval import METRICS, Evaluator, paired_bootstrap  # noqa: E402

CONDITIONS = {
    "base": ("base BGE-M3", os.path.join(ROOT, "cache")),
    "sapt_query_only": ("SAPT queries + base passages", os.path.join(HERE, "cache", "sapt_query_only")),
    "sapt_symmetric": ("SAPT queries + SAPT passages", os.path.join(HERE, "cache", "sapt_symmetric")),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "zero_shot_{split}.json"))
    args = ap.parse_args()

    per_query, report, emb, ref_qids = {}, {}, {}, None
    passage_rows = torch.randint(0, 500_000, (5000,), generator=torch.Generator().manual_seed(0))
    for key, (label, cache_dir) in CONDITIONS.items():
        if not os.path.exists(os.path.join(cache_dir, "corpus_emb.npy")):
            print(f"skipping {key}: no cache at {cache_dir}")
            continue
        store = EmbeddingStore(cache_dir)
        with open(os.path.join(cache_dir, "splits.json")) as f:
            ev = Evaluator(store, json.load(f)[args.split], args.data_dir)
        ref_qids = ref_qids or ev.qids
        assert ev.qids == ref_qids, f"{key}: evaluated queries differ"
        z_form, z_inf = ev.formal(), ev.informal()
        pq = {"formal": ev.run_per_query(z_form), "informal": ev.run_per_query(z_inf)}
        per_query[key] = pq
        means = {c: {m: float(pq[c][m].mean()) for m in METRICS} for c in pq}
        report[key] = {"label": label, **means, "drop": {m: means["informal"][m] - means["formal"][m] for m in METRICS},
                       "cos_informal_formal_twin": float((z_inf * z_form).sum(-1).mean())}
        emb[key] = {"formal": z_form.cpu(), "informal": z_inf.cpu(),
                    "passages": store.corpus[passage_rows.to(store.device)].float().cpu()}
        del store, ev
        torch.cuda.empty_cache()

    if "base" in emb:
        for key in report:
            if key == "base":
                continue
            report[key]["drift_vs_base"] = {
                "queries_formal": float((emb[key]["formal"] * emb["base"]["formal"]).sum(-1).mean()),
                "queries_informal": float((emb[key]["informal"] * emb["base"]["informal"]).sum(-1).mean()),
                "passages_5k_sample": float((emb[key]["passages"] * emb["base"]["passages"]).sum(-1).mean()),
            }

    print(f"\nZero-shot, MIRACL-id {args.split} split: {len(ref_qids)} queries, 500,000 passages (no adapter)")
    print(f"{'condition':<30} | {'formal nDCG@10':>14} {'MRR@10':>7} {'R@100':>7} | {'informal nDCG@10':>16} {'MRR@10':>7} {'R@100':>7} | {'drop':>7} | cos(inf,formal)")
    for key, r in report.items():
        f, i = r["formal"], r["informal"]
        print(f"{r['label']:<30} | {f['nDCG@10']:>14.4f} {f['MRR@10']:>7.4f} {f['Recall@100']:>7.4f} | {i['nDCG@10']:>16.4f} "
              f"{i['MRR@10']:>7.4f} {i['Recall@100']:>7.4f} | {100 * r['drop']['nDCG@10']:>+6.2f} | {r['cos_informal_formal_twin']:.4f}")
    for key, r in report.items():
        if "drift_vs_base" in r:
            d = r["drift_vs_base"]
            print(f"  drift of {r['label']} vs base BGE-M3 (mean cosine to its own base vector): formal queries "
                  f"{d['queries_formal']:.4f} | informal queries {d['queries_informal']:.4f} | passages (5k sample) {d['passages_5k_sample']:.4f}")

    significance = {}
    if "base" in per_query:
        print(f"\nPaired bootstrap vs base BGE-M3 ({args.bootstrap} samples, two-sided; * = p < 0.05)")
        for key in per_query:
            if key == "base":
                continue
            for qtype in ("formal", "informal"):
                label = f"{report[key]['label']} - base ({qtype})"
                significance[label] = {m: paired_bootstrap(per_query[key][qtype][m], per_query["base"][qtype][m], args.bootstrap) for m in METRICS}
            drop = {m: per_query[key]["informal"][m] - per_query[key]["formal"][m] for m in METRICS}
            base_drop = {m: per_query["base"]["informal"][m] - per_query["base"]["formal"][m] for m in METRICS}
            significance[f"{report[key]['label']} - base (informal drop)"] = {m: paired_bootstrap(drop[m], base_drop[m], args.bootstrap) for m in METRICS}
        cw = max(len(k) for k in significance) + 2
        for label, res in significance.items():
            for m, r in res.items():
                ci = f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
                print(f"  {label:<{cw}} {m:<11} {r['diff']:>+8.4f} {ci:>20} {r['p']:>7.3f}{' *' if r['p'] < 0.05 else ''}")

    out = args.out.format(split=args.split)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"split": args.split, "n_queries": len(ref_qids), "conditions": report,
                   "bootstrap_samples": args.bootstrap, "significance": significance}, f, indent=2)
    print(f"\n-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
