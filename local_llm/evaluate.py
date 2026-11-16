"""Evaluate local LLM query normalization against base encoder and trained adapters.

Encodes the LLM-normalized queries with the frozen BGE-M3 encoder, searches the cached corpus and
compares with paired bootstrap tests:
  1. base BGE-M3 on formal_query                  (upper bound)
  2. base BGE-M3 on generated_informal_query       (zero-shot)
  3. base BGE-M3 on LLM-normalised informal query  (local LLM rewrite baseline)
  4. Query-DANN adapted embeddings (optional, with --run)

Usage:
  python3 local_llm/evaluate.py --rewrites local_llm/cache/qwen2.5_3b-instruct-q4_K_M_test.jsonl \
      --run query_dann_v2=methods/query_dann/outputs/v2_dann_g1.0/best
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
ROOT = os.path.dirname(HERE)  # bge_dann root
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "methods", "query_distillation"))

from preprocessing.dataset import EmbeddingStore, load_train_pairs  # noqa: E402
from methods.eval import METRICS, Evaluator, paired_bootstrap  # noqa: E402
from methods.query_distillation.evaluate import adapt, load_adapter  # noqa: E402
from methods.model import BGEM3Encoder  # noqa: E402


FORMAL = "1. base BGE-M3 | formal query (upper bound)"
ZERO_SHOT = "2. base BGE-M3 | informal query (zero-shot)"
REWRITE = "3. base BGE-M3 | local LLM-normalised query"


def load_rewrites(rewrites_path: str, evaluated_qids: List[str]) -> Dict[str, Dict]:
    """Load rewrites from JSONL, filter to evaluated queries."""
    rows = {}
    with open(rewrites_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["query_id"]] = r
    
    missing = [q for q in evaluated_qids if q not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} evaluated queries have no rewrite (e.g. {missing[:5]}); "
                         f"re-run local_llm/normalize.py --split {args.split}")
    
    return rows


def main():
    global args
    ap = argparse.ArgumentParser(description="Evaluate local LLM query normalization")
    ap.add_argument("--rewrites", required=True, help="JSONL produced by local_llm/normalize.py")
    ap.add_argument("--run", action="append", default=[], help="name=checkpoint_dir (repeatable)")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "methods", "cache"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Load rewrites
    with open(args.rewrites) as f:
        first_line = f.readline().strip()
        first_row = json.loads(first_line) if first_line else {}
    
    rewrite_model = first_row.get("model", "unknown")
    prompt_version = first_row.get("prompt_version", "unknown")
    shots = first_row.get("shots", 0)
    batch_sizes = set()
    
    # Load all rows
    rows = {}
    with open(args.rewrites) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["query_id"]] = r
            if "batch_size" in r:
                batch_sizes.add(r["batch_size"])

    # Setup evaluator
    store = EmbeddingStore(args.cache_dir)
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        splits = json.load(f)
    ev = Evaluator(store, splits[args.split], args.data_dir)

    missing = [q for q in ev.qids if q not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} evaluated queries have no rewrite (e.g. {missing[:5]}); "
                         f"re-run local_llm/normalize.py --split {args.split}")

    # Live encoding of rewritten queries with frozen BGE-M3 (max_length=64 as in prepare.py)
    encoder = BGEM3Encoder(args.model)
    texts = [rows[q]["normalized"] for q in ev.qids]
    t0 = time.monotonic()
    z = torch.from_numpy(encoder.encode(texts, max_length=64, desc="rewritten queries")).to(store.device).float()
    encode_s = (time.monotonic() - t0) / len(texts)
    del encoder
    torch.cuda.empty_cache()

    # Compute per-query metrics
    per_query = {
        FORMAL: ev.run_per_query(ev.formal()),
        ZERO_SHOT: ev.run_per_query(ev.informal()),
        REWRITE: ev.run_per_query(z),
    }
    
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

    # Aggregate results
    results = {row: {m: float(v.mean()) for m, v in pq.items()} for row, pq in per_query.items()}

    # Print results table
    upper, lower = results[FORMAL]["nDCG@10"], results[ZERO_SHOT]["nDCG@10"]
    width = max(len(r) for r in results) + 2
    print(f"\nMIRACL-id {args.split}: {len(ev.qids)} queries, {len(store.doc_ids):,} passages")
    print(f"{'condition':<{width}} {'nDCG@10':>8} {'MRR@10':>8} {'R@100':>8} {'gap closed':>11}")
    for row, m in results.items():
        gap = "" if row == FORMAL else f"{100 * (m['nDCG@10'] - lower) / (upper - lower):>10.1f}%"
        print(f"{row:<{width}} {m['nDCG@10']:>8.4f} {m['MRR@10']:>8.4f} {m['Recall@100']:>8.4f} {gap:>11}")

    # Paired bootstrap significance tests
    comparisons = [
        ("local LLM rewrite vs zero-shot", REWRITE, ZERO_SHOT),
        ("local LLM rewrite vs formal", REWRITE, FORMAL),
    ]
    for spec in args.run:
        name = spec.split("=", 1)[0]
        comparisons.append((f"local LLM rewrite vs {name}", REWRITE, f"   {name} | informal query"))
    
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

    # Rewrite statistics
    tp = load_train_pairs(args.data_dir).set_index("query_id")
    norm_func = lambda s: " ".join(s.lower().replace("?", " ").split())
    same = sum(1 for q in ev.qids if norm_func(rows[q]["normalized"]) == norm_func(tp.at[q, "formal_query"]))
    lat = sorted(rows[q]["latency_s"] for q in ev.qids)
    tok = [rows[q].get("total_tokens", 0) for q in ev.qids if rows[q].get("total_tokens", 0) > 0]
    
    print(f"\nRewrite model: {rewrite_model} (prompt {prompt_version}, {shots}-shot, batch sizes {sorted(batch_sizes)})")
    print(f"  identical to the original formal query: {same}/{len(ev.qids)} ({100 * same / len(ev.qids):.1f}%)")
    print(f"  LLM latency per query: mean {sum(lat) / len(lat):.2f}s p50 {lat[len(lat) // 2]:.2f}s "
          f"p95 {lat[int(0.95 * len(lat))]:.2f}s" + ("  (batched: per-query latency is amortised)" if batch_sizes != {1} else ""))
    if tok:
        print(f"  tokens per query: mean {sum(tok)/len(tok):.1f}")
    print(f"  query encoding (BGE-M3, needed by every condition): {1000 * encode_s:.1f} ms/query")
    for name, ms in adapter_ms.items():
        speedup = (sum(lat) / len(lat) * 1000) / max(ms, 1e-9)
        print(f"  {name} adapter forward: {ms:.4f} ms/query  -> ~{speedup:.0f}x faster than the LLM call")

    # Save results
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "split": args.split,
                "n_queries": len(ev.qids),
                "rewrites": args.rewrites,
                "rewrite_model": rewrite_model,
                "prompt_version": prompt_version,
                "shots": shots,
                "batch_sizes": sorted(batch_sizes),
                "results": results,
                "significance": significance,
                "bootstrap_samples": args.bootstrap,
                "identical_to_formal": {"count": same, "pct": 100 * same / len(ev.qids)},
                "llm_latency_s": {
                    "mean": sum(lat) / len(lat),
                    "p50": lat[len(lat) // 2],
                    "p95": lat[int(0.95 * len(lat))],
                },
                "llm_tokens_per_query": {
                    "mean": sum(tok) / len(tok) if tok else 0,
                },
                "encode_ms_per_query": 1000 * encode_s,
                "adapter_ms_per_query": adapter_ms,
            }, f, indent=2)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()