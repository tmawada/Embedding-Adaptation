"""Retrieval evaluation on MIRACL-id: nDCG@10, MRR@10, Recall@100 + paired bootstrap tests.

Conditions (test split by default):
  1. Base BGE-M3 on formal_query                 (upper bound)
  2. Base BGE-M3 on generated_informal_query     (zero-shot baseline)
  3. Query-DANN adapted BGE-M3 on informal query (proposed)
Extra rows: the same adapter applied to formal queries (does it break clean
input?), and any additional checkpoints passed via --extra (e.g. gamma=0 ablation).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import torch

from dataset import CACHE_DIR, DATA_DIR, EmbeddingStore, load_qrels
from model import QueryDANN

METRICS = ("nDCG@10", "MRR@10", "Recall@100")


@torch.no_grad()
def dense_search(queries: torch.Tensor, corpus: torch.Tensor, k: int = 100, batch_size: int = 256):
    """Exact inner-product (= cosine, all vectors L2-normalised) top-k over the corpus."""
    scores, idx = [], []
    if corpus.device.type == "cpu":  # no half-precision matmul on CPU
        corpus = corpus.float()
    for i in range(0, len(queries), batch_size):
        s = queries[i : i + batch_size].to(corpus.dtype) @ corpus.T
        top = s.float().topk(k, dim=1)
        scores.append(top.values.cpu())
        idx.append(top.indices.cpu())
    return torch.cat(scores), torch.cat(idx)


def per_query_metrics(ranked: Dict[str, List[str]], qrels: Dict[str, set]) -> Dict[str, np.ndarray]:
    """Binary-relevance nDCG@10, MRR@10, Recall@100 per query, in the iteration order of `ranked`."""
    out = {m: [] for m in METRICS}
    for qid, docs in ranked.items():
        rel = qrels[qid]
        hits = [d in rel for d in docs]
        dcg = sum(1.0 / math.log2(r + 2) for r, h in enumerate(hits[:10]) if h)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(rel), 10)))
        out["nDCG@10"].append(dcg / idcg)
        out["MRR@10"].append(next((1.0 / (r + 1) for r, h in enumerate(hits[:10]) if h), 0.0))
        out["Recall@100"].append(sum(hits[:100]) / len(rel))
    return {m: np.array(v) for m, v in out.items()}


def retrieval_metrics(ranked: Dict[str, List[str]], qrels: Dict[str, set]) -> Dict[str, float]:
    return {m: float(v.mean()) for m, v in per_query_metrics(ranked, qrels).items()}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_samples: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap over queries for mean(a - b).

    ci95: percentile interval of the resampled mean difference.
    p: two-sided p-value from the null-shifted bootstrap (differences re-centred to mean 0),
       with the (count + 1) / (n + 1) correction so p is never reported as exactly 0.
    """
    d = a - b
    observed = d.mean()
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_samples, len(d)))
    boot = d[idx].mean(axis=1)
    null = (d - observed)[idx].mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_samples + 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(observed), "ci95_low": float(lo), "ci95_high": float(hi), "p": float(p)}


class Evaluator:
    def __init__(self, store: EmbeddingStore, qids: List[str], data_dir: str = DATA_DIR):
        self.store = store
        # Relevance restricted to docs present in the corpus (others are unretrievable).
        in_corpus = set(store.doc2row)
        full = load_qrels(data_dir)
        self.qrels = {q: full[q] & in_corpus for q in qids if full.get(q, set()) & in_corpus}
        self.qids = sorted(self.qrels, key=int)
        self.rows = torch.tensor([store.qid2row[q] for q in self.qids], device=store.device)

    def _ranked(self, z_queries: torch.Tensor, k: int):
        _, idx = dense_search(z_queries, self.store.corpus, k)
        ids = self.store.doc_ids
        return {q: [ids[j] for j in row] for q, row in zip(self.qids, idx.tolist())}

    def run(self, z_queries: torch.Tensor, k: int = 100) -> Dict[str, float]:
        return retrieval_metrics(self._ranked(z_queries, k), self.qrels)

    def run_per_query(self, z_queries: torch.Tensor, k: int = 100) -> Dict[str, np.ndarray]:
        return per_query_metrics(self._ranked(z_queries, k), self.qrels)

    def formal(self):
        return self.store.formal[self.rows].float()

    def informal(self):
        return self.store.informal[self.rows].float()

    @torch.no_grad()
    def adapted(self, head: QueryDANN, which: str = "informal"):
        was_training = head.training
        head.eval()
        z = head.adapt(self.informal() if which == "informal" else self.formal())
        head.train(was_training)
        return z


def load_head(ckpt_dir: str, device) -> QueryDANN:
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    # Checkpoints from the first run predate the adapter/discriminator fields.
    head = QueryDANN(cfg["bottleneck"], cfg["adapter_dropout"], cfg["disc_dropout"],
                     cfg.get("adapter", "layernorm"), cfg.get("discriminator", "plain")).to(device)
    head.adapter.load_state_dict(torch.load(os.path.join(ckpt_dir, "adapter.pt"), map_location=device))
    head.discriminator.load_state_dict(torch.load(os.path.join(ckpt_dir, "discriminator.pt"), map_location=device))
    return head.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Query-DANN checkpoint dir (runs/<name>/best)")
    ap.add_argument("--extra", nargs="*", default=[], help="Additional name=ckpt_dir rows, e.g. infonce_only=runs/g0/best")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--bootstrap", type=int, default=1000, help="Paired bootstrap samples (0 = skip significance tests)")
    ap.add_argument("--cache_dir", default=CACHE_DIR)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--out", default=None, help="Write results JSON here")
    args = ap.parse_args()

    store = EmbeddingStore(args.cache_dir)
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        qids = json.load(f)[args.split]
    ev = Evaluator(store, qids, args.data_dir)
    head = load_head(args.ckpt, store.device)

    base_formal = "1. Base BGE-M3 | formal query (upper bound)"
    zero_shot = "2. Base BGE-M3 | informal query (zero-shot)"
    proposed = "3. Query-DANN  | informal query (proposed)"
    robust = "   Query-DANN  | formal query (robustness)"
    per_query = {
        base_formal: ev.run_per_query(ev.formal()),
        zero_shot: ev.run_per_query(ev.informal()),
        proposed: ev.run_per_query(ev.adapted(head, "informal")),
        robust: ev.run_per_query(ev.adapted(head, "formal")),
    }
    comparisons = [(f"DANN informal vs zero-shot informal", proposed, zero_shot),
                   (f"DANN formal vs base formal", robust, base_formal)]
    for spec in args.extra:
        name, path = spec.split("=", 1)
        row = f"   {name} | informal query"
        per_query[row] = ev.run_per_query(ev.adapted(load_head(path, store.device), "informal"))
        comparisons.insert(1, (f"DANN informal vs {name} informal", proposed, row))
    results = {name: {m: float(v.mean()) for m, v in pq.items()} for name, pq in per_query.items()}

    print(f"\nMIRACL-id {args.split} split: {len(ev.qids)} queries, {len(store.doc_ids):,} passages")
    print(f"{'condition':<48} {'nDCG@10':>8} {'MRR@10':>8} {'R@100':>8}")
    for name, m in results.items():
        print(f"{name:<48} {m['nDCG@10']:>8.4f} {m['MRR@10']:>8.4f} {m['Recall@100']:>8.4f}")

    significance = {}
    if args.bootstrap > 0:
        print(f"\nPaired bootstrap ({args.bootstrap} samples, two-sided; * = p < 0.05)")
        print(f"{'comparison':<40} {'metric':<11} {'diff':>8} {'95% CI':>20} {'p':>7}")
        for label, a, b in comparisons:
            significance[label] = {}
            for m in METRICS:
                r = paired_bootstrap(per_query[a][m], per_query[b][m], args.bootstrap)
                significance[label][m] = r
                ci = f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
                print(f"{label:<40} {m:<11} {r['diff']:>+8.4f} {ci:>20} {r['p']:>7.3f}{' *' if r['p'] < 0.05 else ''}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "n_queries": len(ev.qids), "ckpt": args.ckpt,
                       "results": results, "bootstrap_samples": args.bootstrap, "significance": significance}, f, indent=2)


if __name__ == "__main__":
    main()
