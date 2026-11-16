"""Zero-shot evaluation of the InfoNCE-tuned tower against base BGE-M3 — no adapter involved.

Conditions (same test queries, same 500k passages, same truncation length):
  base            original BGE-M3 for queries and passages            (../cache)
  infc_query_only InfoNCE-tuned queries vs original passage embeddings (cache/infc_query_only)

Retrieval only: formal / informal nDCG@10, MRR@10, Recall@100, the informal drop, and the paired
bootstrap against base. Embedding diagnostics (twin cosine, drift vs base) are deliberately not part of
this path — ../tsdae_sapt/evaluate_sapt.py still reports them if the mechanism needs checking.

  The FORMAL column is the number to watch. ../tsdae_sapt failed on this method: its formal nDCG@10 fell
  0.6166 -> 0.5340, because pure reconstruction never anchored the encoder to retrieval. A formal column
  that collapses the same way here means the preserve term (query_dann/train.py) is the fix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from preprocessing.dataset import EmbeddingStore  # noqa: E402
from methods.eval import METRICS, Evaluator, paired_bootstrap  # noqa: E402

CONDITIONS = {
    "base": ("base BGE-M3", os.path.join(ROOT, "methods", "cache")),
    "infc_query_only": ("InfoNCE queries + base passages", os.path.join(HERE, "cache", "infc_query_only")),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "zero_shot_{split}.json"))
    args = ap.parse_args()

    per_query, report, ref_qids = {}, {}, None
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
        report[key] = {"label": label, **means, "drop": {m: means["informal"][m] - means["formal"][m] for m in METRICS}}
        del store, ev
        torch.cuda.empty_cache()

    mw = max((len(r["label"]) for r in report.values()), default=len("Model"))
    w = mw + 58  # label column + " | " + 26 + " | " + 26 — same layout as the training tables

    # ---- retrieval metrics, in the table style used by the training runs ----
    print(f"\n{'=' * w}")
    print(f"Zero-shot, MIRACL-id {args.split} split: {len(ref_qids)} queries, 500,000 passages (no adapter)")
    print(f"{'=' * w}")
    print(f"{'Model':<{mw}} | {'Formal':^26} | {'Informal':^26}")
    print(f"{'':<{mw}} | {'R@100':>8} {'MRR@10':>8} {'nDCG@10':>8} | {'R@100':>8} {'MRR@10':>8} {'nDCG@10':>8}")
    print(f"{'-' * w}")
    for r in report.values():
        form, inf = r["formal"], r["informal"]
        print(f"{r['label']:<{mw}} | {form['Recall@100']:>8.4f} {form['MRR@10']:>8.4f} {form['nDCG@10']:>8.4f} | "
              f"{inf['Recall@100']:>8.4f} {inf['MRR@10']:>8.4f} {inf['nDCG@10']:>8.4f}")
    print(f"{'=' * w}")

    significance = {}
    if "base" in per_query:
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
        wb = cw + 57  # label + " | " + metric + " " + diff + " | " + ci + " | " + p
        print(f"\n{'=' * wb}")
        print(f"Paired bootstrap vs base BGE-M3 ({args.bootstrap} samples, two-sided; * = p < 0.05)")
        print(f"{'=' * wb}")
        print(f"{'comparison':<{cw}} | {'metric':<10} {'diff':>8} | {'95% CI':>20} | {'p':>9}")
        print(f"{'-' * wb}")
        for label, res in significance.items():
            for m, r in res.items():
                ci = f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
                p = f"{r['p']:.3f}" + (" *" if r["p"] < 0.05 else "  ")
                print(f"{label:<{cw}} | {m:<10} {r['diff']:>+8.4f} | {ci:>20} | {p:>9}")
        print(f"{'=' * wb}")

    out = args.out.format(split=args.split)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"split": args.split, "n_queries": len(ref_qids), "conditions": report,
                   "bootstrap_samples": args.bootstrap, "significance": significance}, f, indent=2)
    print(f"\n-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
