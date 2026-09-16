"""Evaluate LLM query normalisation against the base encoder and (optionally) the trained adapters.

Encodes the LLM-normalised queries with the frozen BGE-M3 encoder, searches the cached 500k corpus and
compares, on the same queries and with paired bootstrap tests:
  1. base BGE-M3 on formal_query                  (upper bound)
  2. base BGE-M3 on generated_informal_query       (zero-shot)
  3. base BGE-M3 on LLM-normalised informal query  (LLM rewrite baseline)
  +  any adapter checkpoint passed with --run name=dir (e.g. score distillation, Query-DANN v2)

  python3 llm_rewrite/evaluate_rewrite.py --rewrites llm_rewrite/cache/gemini-2.5-flash_test.jsonl \
      --run score_distill=query_distillation/runs/score/best --run query_dann_v2=runs/v2_dann_g1.0/best
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "query_distillation"))

from dataset import EmbeddingStore, load_train_pairs  # noqa: E402
from eval import METRICS, Evaluator, paired_bootstrap  # noqa: E402
from evaluate import adapt, load_adapter  # noqa: E402  (query_distillation/evaluate.py)
from model import BGEM3Encoder  # noqa: E402

FORMAL = "1. base BGE-M3 | formal query (upper bound)"
ZERO_SHOT = "2. base BGE-M3 | informal query (zero-shot)"
REWRITE = "3. base BGE-M3 | LLM-normalised query"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites", required=True, help="JSONL produced by normalize.py")
    ap.add_argument("--run", action="append", default=[], help="name=checkpoint_dir (repeatable)")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.rewrites) as f:
        rows = {r["query_id"]: r for r in map(json.loads, filter(str.strip, f))}
    store = EmbeddingStore(args.cache_dir)
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        ev = Evaluator(store, json.load(f)[args.split], args.data_dir)
    missing = [q for q in ev.qids if q not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} evaluated queries have no rewrite (e.g. {missing[:5]}); "
                         f"re-run normalize.py --split {args.split}")

    # Live encoding of the rewritten queries with the same frozen encoder (max_length as in ../prepare.py).
    encoder = BGEM3Encoder(args.model)
    texts = [rows[q]["normalized"] for q in ev.qids]
    t0 = time.monotonic()
    z = torch.from_numpy(encoder.encode(texts, max_length=64, desc="rewritten queries")).to(store.device).float()
    encode_s = (time.monotonic() - t0) / len(texts)
    del encoder
    torch.cuda.empty_cache()

    per_query = {FORMAL: ev.run_per_query(ev.formal()), ZERO_SHOT: ev.run_per_query(ev.informal()),
                 REWRITE: ev.run_per_query(z)}
    adapter_ms = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        adapter = load_adapter(os.path.join(ROOT, path) if not os.path.isabs(path) else path, store.device)
        zi = ev.informal()
        torch.cuda.synchronize()
        t0 = time.monotonic()
        za = adapt(adapter, zi)
        torch.cuda.synchronize()
        adapter_ms[name] = 1000 * (time.monotonic() - t0) / len(ev.qids)
        per_query[f"   {name} | informal query"] = ev.run_per_query(za)
    results = {row: {m: float(v.mean()) for m, v in pq.items()} for row, pq in per_query.items()}

    upper, lower = results[FORMAL]["nDCG@10"], results[ZERO_SHOT]["nDCG@10"]
    width = max(len(r) for r in results) + 2
    print(f"\nMIRACL-id {args.split}: {len(ev.qids)} queries, {len(store.doc_ids):,} passages")
    print(f"{'condition':<{width}} {'nDCG@10':>8} {'MRR@10':>8} {'R@100':>8} {'gap closed':>11}")
    for row, m in results.items():
        gap = "" if row == FORMAL else f"{100 * (m['nDCG@10'] - lower) / (upper - lower):>10.1f}%"
        print(f"{row:<{width}} {m['nDCG@10']:>8.4f} {m['MRR@10']:>8.4f} {m['Recall@100']:>8.4f} {gap:>11}")

    comparisons = [("LLM rewrite vs zero-shot", REWRITE, ZERO_SHOT), ("LLM rewrite vs formal", REWRITE, FORMAL)]
    for spec in args.run:
        name = spec.split("=", 1)[0]
        comparisons.append((f"LLM rewrite vs {name}", REWRITE, f"   {name} | informal query"))
    significance = {}
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

    tp = load_train_pairs(args.data_dir).set_index("query_id")
    norm = lambda s: " ".join(s.lower().replace("?", " ").split())
    same = sum(1 for q in ev.qids if norm(rows[q]["normalized"]) == norm(tp.at[q, "formal_query"]))
    lat = sorted(rows[q]["latency_s"] for q in ev.qids)
    batched = {rows[q].get("batch_size", 1) for q in ev.qids}
    print(f"\nRewrite model: {rows[ev.qids[0]]['model']} (prompt {rows[ev.qids[0]]['prompt_version']}, "
          f"{rows[ev.qids[0]]['shots']}-shot, batch sizes {sorted(batched)})")
    print(f"  identical to the original formal query: {same}/{len(ev.qids)} ({100 * same / len(ev.qids):.1f}%)")
    print(f"  LLM latency per query: mean {sum(lat) / len(lat):.2f}s p50 {lat[len(lat) // 2]:.2f}s "
          f"p95 {lat[int(0.95 * len(lat))]:.2f}s" + ("  (batched: per-query latency is amortised)" if batched != {1} else ""))
    print(f"  query encoding (BGE-M3, needed by every condition): {1000 * encode_s:.1f} ms/query")
    for name, ms in adapter_ms.items():
        print(f"  {name} adapter forward: {ms:.4f} ms/query  -> ~{sum(lat) / len(lat) / max(ms, 1e-9) * 1000:.0f}x faster than the LLM call")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "n_queries": len(ev.qids), "rewrites": args.rewrites,
                       "rewrite_model": rows[ev.qids[0]]["model"], "results": results,
                       "significance": significance, "bootstrap_samples": args.bootstrap,
                       "identical_to_formal": {"count": same, "pct": 100 * same / len(ev.qids)},
                       "llm_latency_s": {"mean": sum(lat) / len(lat), "p50": lat[len(lat) // 2], "p95": lat[int(0.95 * len(lat))]},
                       "encode_ms_per_query": 1000 * encode_s, "adapter_ms_per_query": adapter_ms}, f, indent=2)


if __name__ == "__main__":
    main()
