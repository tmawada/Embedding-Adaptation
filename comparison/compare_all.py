"""One table: BGE-M3 base vs query adapters vs TSDAE-SAPT (two variants) vs LLM normalisation.

Every row is BGE-M3 **dense** retrieval over the same 500k passage index and the same held-out queries,
so all pairwise paired-bootstrap tests are valid.

  1 base BGE-M3, formal queries                    upper bound
  2 base BGE-M3, informal queries                  zero-shot lower bound
  3 BGE-M3 + LLM normalisation                     informal -> formal text, then the untouched encoder
  4 Query-DANN v2 / query distillation             adapters on the frozen base tower
  5 TSDAE-SAPT tower, no adapter                   stage 1 alone, for each SAPT variant
  6 TSDAE-SAPT + Query-DANN / + distillation       the ADAPT-DANN stack (stage 1 + stage 2)

SAPT variants:
  reviews  tsdae_sapt/          TSDAE on 4,911 informal Indonesian reviews (prdect-id)
  queries  tsdae_sapt_queries/  TSDAE on the task's own 2,891 informal train queries (data ablation)

  python3 comparison/compare_all.py
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
from eval import METRICS, Evaluator, paired_bootstrap  # noqa: E402
from evaluate import adapt, load_adapter  # noqa: E402
from model import BGEM3Encoder  # noqa: E402

FORMAL = "BGE-M3 base - formal query (upper bound)"
ZERO = "BGE-M3 base - informal query (zero-shot)"


def abspath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def query_vectors(cache_dir: str, kind: str, qids, device) -> torch.Tensor:
    """formal/informal query embeddings for `qids` from any cache sharing the base query order."""
    with open(os.path.join(cache_dir, "query_ids.json")) as f:
        order = {q: i for i, q in enumerate(json.load(f))}
    emb = np.load(os.path.join(cache_dir, f"{kind}_emb.npy"), mmap_mode="r")
    rows = [order[q] for q in qids]
    return F.normalize(torch.from_numpy(np.asarray(emb[rows], dtype=np.float32)).to(device), dim=-1)


def best_dann(dir_path: str):
    """Checkpoint with the best dev nDCG@10 among gamma>0 runs of a Query-DANN sweep."""
    if not os.path.isdir(dir_path):
        return None
    cands = [os.path.join(dir_path, d, "best") for d in os.listdir(dir_path) if d != "g0"]
    cands = [c for c in cands if os.path.exists(os.path.join(c, "config.json"))]
    return max(cands, key=lambda c: json.load(open(os.path.join(c, "config.json")))["dev"]["nDCG@10"]) if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--rewrites", default="llm_rewrite/cache/gemini-3.5-flash-lite_test.jsonl")
    ap.add_argument("--dann", default="runs/v2_dann_g1.0/best")
    ap.add_argument("--distill_score", default="query_distillation/runs/score/best")
    ap.add_argument("--distill_embed", default="query_distillation/runs/embed/best")
    # SAPT variant A: informal reviews (prdect-id)
    ap.add_argument("--sapt_reviews_cache", default="tsdae_sapt/cache/sapt_query_only")
    ap.add_argument("--sapt_reviews_symmetric", default="tsdae_sapt/cache/sapt_symmetric")
    ap.add_argument("--sapt_reviews_dann", default="encoder_comparison/runs/sapt_reviews_bge_m3/query_dann_v2")
    ap.add_argument("--sapt_reviews_distill", default="encoder_comparison/runs/sapt_reviews_bge_m3/query_distillation/score/best")
    # SAPT variant B: the task's own informal queries
    ap.add_argument("--sapt_queries_cache", default="tsdae_sapt_queries/runs/tsdae_informal/cache")
    ap.add_argument("--sapt_queries_dann", default="encoder_comparison/runs/tsdae_bge_m3/query_dann_v2")
    ap.add_argument("--sapt_queries_distill", default="encoder_comparison/runs/tsdae_bge_m3/query_distillation/score/best")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "runs"))
    args = ap.parse_args()

    store = EmbeddingStore(args.cache_dir)
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        ev = Evaluator(store, json.load(f)[args.split], args.data_dir)
    device = store.device
    pq, notes = {}, {}

    pq[FORMAL] = ev.run_per_query(ev.formal())
    pq[ZERO] = ev.run_per_query(ev.informal())

    rewrites = abspath(args.rewrites)
    if os.path.exists(rewrites):
        with open(rewrites) as f:
            rows = {r["query_id"]: r for r in map(json.loads, filter(str.strip, f))}
        if [q for q in ev.qids if q not in rows]:
            print(f"! skipping LLM normalisation: rewrites missing for some {args.split} queries")
        else:
            enc = BGEM3Encoder()
            z = torch.from_numpy(enc.encode([rows[q]["normalized"] for q in ev.qids], max_length=64,
                                            desc="LLM-normalised queries")).to(device).float()
            del enc
            torch.cuda.empty_cache()
            notes["llm_model"] = rows[ev.qids[0]]["model"]
            pq[f"BGE-M3 + LLM normalisation ({notes['llm_model']})"] = ev.run_per_query(z)
    else:
        print(f"! no rewrites at {args.rewrites}")

    for label, ckpt in [("Query-DANN v2 (adapter on base tower)", args.dann),
                        ("Query distillation: score (adapter on base tower)", args.distill_score),
                        ("Query distillation: embed (adapter on base tower)", args.distill_embed)]:
        p = abspath(ckpt)
        if os.path.exists(os.path.join(p, "adapter.pt")):
            pq[label] = ev.run_per_query(adapt(load_adapter(p, device), ev.informal()))
        else:
            print(f"! missing checkpoint {ckpt}")

    variants = [("TSDAE-SAPT (reviews)", args.sapt_reviews_cache, args.sapt_reviews_dann, args.sapt_reviews_distill),
                ("TSDAE-SAPT (task queries)", args.sapt_queries_cache, args.sapt_queries_dann, args.sapt_queries_distill)]
    for label, cache, dann_dir, distill_ckpt in variants:
        cache = abspath(cache)
        if not os.path.exists(os.path.join(cache, "informal_emb.npy")):
            print(f"! no SAPT cache at {cache}")
            continue
        z_inf = query_vectors(cache, "informal", ev.qids, device)
        pq[f"{label} tower - informal query (no adapter)"] = ev.run_per_query(z_inf)
        pq[f"{label} tower - formal query"] = ev.run_per_query(query_vectors(cache, "formal", ev.qids, device))
        ck = best_dann(abspath(dann_dir))
        if ck:
            pq[f"{label} + Query-DANN v2 (full stack)"] = ev.run_per_query(adapt(load_adapter(ck, device), z_inf))
            notes[f"{label}_dann_ckpt"] = os.path.relpath(ck, ROOT)
        d = abspath(distill_ckpt)
        if os.path.exists(os.path.join(d, "adapter.pt")):
            pq[f"{label} + distillation: score"] = ev.run_per_query(adapt(load_adapter(d, device), z_inf))

    sym = abspath(args.sapt_reviews_symmetric)
    if os.path.exists(os.path.join(sym, "corpus_emb.npy")) and not os.path.islink(os.path.join(sym, "corpus_emb.npy")):
        sym_store = EmbeddingStore(sym)
        sym_ev = Evaluator(sym_store, ev.qids, args.data_dir)
        pq["SAPT-reviews symmetric (SAPT queries AND passages)"] = sym_ev.run_per_query(sym_ev.informal())
        notes["symmetric_note"] = "re-encodes the 500k index with the SAPT tower (index rebuild required)"
        del sym_store, sym_ev
        torch.cuda.empty_cache()

    results = {k: {m: float(v.mean()) for m, v in a.items()} for k, a in pq.items()}
    upper, lower = results[FORMAL]["nDCG@10"], results[ZERO]["nDCG@10"]
    width = max(len(k) for k in results) + 2
    lines = [f"# Informal-query retrieval on MIRACL-id {args.split} ({len(ev.qids)} queries, {len(store.doc_ids):,} passages)",
             "", "All rows: BGE-M3 **dense** retrieval. Gap closed = share of the "
             f"{100 * (upper - lower):.2f}-point formal/informal nDCG@10 gap recovered. "
             f"Δ = nDCG@10 points vs zero-shot; * = p < 0.05 (paired bootstrap, {args.bootstrap} samples).", "",
             "| System | nDCG@10 | MRR@10 | R@100 | Gap closed | Δ vs zero-shot |", "|---|---|---|---|---|---|"]
    print(f"\n{'system':<{width}} {'nDCG@10':>8} {'MRR@10':>8} {'R@100':>8} {'gap':>8}  delta vs zero-shot")
    significance = {}
    for name, m in results.items():
        skip = name in (FORMAL, ZERO) or "- formal query" in name
        gap = "" if skip else f"{100 * (m['nDCG@10'] - lower) / (upper - lower):.1f}%"
        delta = ""
        if name not in (FORMAL, ZERO):
            significance[f"{name} vs zero-shot"] = {mm: paired_bootstrap(pq[name][mm], pq[ZERO][mm], args.bootstrap)
                                                    for mm in METRICS}
            r = significance[f"{name} vs zero-shot"]["nDCG@10"]
            delta = f"{100 * r['diff']:+.2f}{'*' if r['p'] < 0.05 else ''} (p={r['p']:.3f})"
        print(f"{name:<{width}} {m['nDCG@10']:>8.4f} {m['MRR@10']:>8.4f} {m['Recall@100']:>8.4f} {gap:>8}  {delta}")
        lines.append(f"| {name} | {m['nDCG@10']:.4f} | {m['MRR@10']:.4f} | {m['Recall@100']:.4f} | {gap} | {delta} |")

    base_adapters = [k for k in results if "adapter on base tower" in k]
    heads = [(a, b) for a in results if a.startswith(("BGE-M3 + LLM", "SAPT")) and "- formal query" not in a
             for b in base_adapters]
    heads += [(f"{v} + Query-DANN v2 (full stack)", f"{v} tower - informal query (no adapter)")
              for v in ("TSDAE-SAPT (reviews)", "TSDAE-SAPT (task queries)")]
    heads = [(a, b) for a, b in heads if a in results and b in results]
    if heads:
        lines += ["", "## Head-to-head (nDCG@10)", "", "| A | B | A - B | p |", "|---|---|---|---|"]
        print("\nhead-to-head (nDCG@10):")
        for a, b in heads:
            r = paired_bootstrap(pq[a]["nDCG@10"], pq[b]["nDCG@10"], args.bootstrap)
            significance[f"{a} vs {b}"] = {"nDCG@10": r}
            print(f"  {a} - {b}: {100 * r['diff']:+.2f}{'*' if r['p'] < 0.05 else ''} (p={r['p']:.3f})")
            lines.append(f"| {a} | {b} | {100 * r['diff']:+.2f}{'*' if r['p'] < 0.05 else ''} | {r['p']:.3f} |")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"comparison_{args.split}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out_dir, f"comparison_{args.split}.json"), "w") as f:
        json.dump({"split": args.split, "n_queries": len(ev.qids), "results": results,
                   "significance": significance, "notes": notes, "bootstrap_samples": args.bootstrap}, f, indent=2)
    print(f"\n-> {os.path.relpath(os.path.join(args.out_dir, f'comparison_{args.split}.md'), ROOT)}")


if __name__ == "__main__":
    main()
