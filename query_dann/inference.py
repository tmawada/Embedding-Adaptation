"""Query-DANN inference demo: live BGE-M3 query encoding + trained adapter, search over the cached corpus.

Sampled mode (default): draws N queries from a split and, for each, reports
  - cosine similarities before / after adaptation (informal<->formal, informal<->gold passage)
  - rank of the first relevant passage for base informal, adapted informal and base formal
  - top-k passages for base vs adapted informal queries (relevant ones marked)
Free-text mode: --query "..." [--query "..."] shows base vs adapted top-k for your own queries.

  python3 inference.py --ckpt runs/v2_dann_g1.0/best --n 5
  python3 inference.py --ckpt runs/v2_dann_g1.0/best --query "presiden pertama indonesia siapa sih"
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import random
import textwrap
from typing import List

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset import CACHE_DIR, DATA_DIR, EmbeddingStore, load_corpus, load_qrels, load_train_pairs
from eval import METRICS, dense_search, load_head, per_query_metrics
from model import BGEM3Encoder

SEARCH_DEPTH = 100


class QueryDANNRetriever:
    """Frozen BGE-M3 query encoder + Query-DANN adapter over pre-encoded corpus embeddings."""

    def __init__(self, ckpt: str, cache_dir: str = CACHE_DIR, data_dir: str = DATA_DIR, model_name: str = "BAAI/bge-m3"):
        self.store = EmbeddingStore(cache_dir)
        self.device = self.store.device
        self.encoder = BGEM3Encoder(model_name, device=str(self.device))
        self.head = load_head(ckpt, self.device)
        doc_ids, self.passages = load_corpus(data_dir)
        assert doc_ids == self.store.doc_ids, "corpus.json order differs from the cached corpus embeddings"

    @torch.no_grad()
    def encode(self, texts: List[str], adapt: bool = False) -> torch.Tensor:
        z = torch.from_numpy(self.encoder.encode(texts, max_length=64, progress=False)).to(self.device).float()
        return self.head.adapt(z) if adapt else z

    def search(self, z: torch.Tensor, k: int = SEARCH_DEPTH):
        scores, idx = dense_search(z, self.store.corpus, k)
        return scores.tolist(), idx.tolist()

    def doc_embedding(self, doc_id: str) -> torch.Tensor:
        return self.store.corpus[self.store.doc2row[doc_id]].float()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum())  # all vectors are L2-normalised


def first_relevant_rank(ranked_rows: List[int], relevant: set, doc_ids: List[str]):
    return next((r + 1 for r, j in enumerate(ranked_rows) if doc_ids[j] in relevant), None)


def fmt_rank(rank) -> str:
    return str(rank) if rank else f">{SEARCH_DEPTH}"


def snippet(text: str, width: int = 96) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder=" …")


CONDITION_LABELS = {"base_informal": "base BGE-M3 (informal)", "adapted_informal": "Query-DANN (informal)",
                    "base_formal": "base BGE-M3 (formal)"}


def print_metrics_table(metrics: dict, title: str = "retrieval metrics"):
    print(f"  {title:<34} {'nDCG@10':>8} {'MRR@10':>8} {'R@100':>8}")
    for name, m in metrics.items():
        print(f"    {CONDITION_LABELS[name]:<32} {m['nDCG@10']:>8.4f} {m['MRR@10']:>8.4f} {m['Recall@100']:>8.4f}")


def print_topk(title: str, scores, rows, retriever: QueryDANNRetriever, k: int, relevant: set = None):
    print(f"  {title}")
    for r in range(k):
        doc_id = retriever.store.doc_ids[rows[r]]
        mark = "" if relevant is None else ("✓" if doc_id in relevant else "·")
        print(f"   {r + 1}. {mark} {scores[r]:.3f} [{doc_id:>11}] {snippet(retriever.passages[rows[r]])}")


def sampled_demo(retriever: QueryDANNRetriever, args):
    with open(f"{args.cache_dir}/splits.json") as f:
        split_qids = json.load(f)[args.split]
    qrels = load_qrels(args.data_dir)
    in_corpus = set(retriever.store.doc2row)
    pairs = load_train_pairs(args.data_dir).set_index("query_id")
    candidates = [q for q in split_qids if qrels.get(q, set()) & in_corpus]
    qids = args.qids or random.Random(args.seed).sample(candidates, args.n)

    formal = [pairs.at[q, "formal_query"] for q in qids]
    informal = [pairs.at[q, "generated_informal_query"] for q in qids]
    z_form, z_inf = retriever.encode(formal), retriever.encode(informal)
    z_adapt = retriever.head.adapt(z_inf)

    cached = retriever.store.informal[[retriever.store.qid2row[q] for q in qids]].float()
    print(f"live query encoding vs training cache: max |Δ| = {(z_inf - cached).abs().max().item():.2e}")

    (s_base, r_base), (s_adapt, r_adapt), (_, r_form) = retriever.search(z_inf), retriever.search(z_adapt), retriever.search(z_form)
    doc_ids = retriever.store.doc_ids
    # Per-query nDCG@10 / MRR@10 / Recall@100, computed exactly as in eval.py (arrays aligned with qids).
    sample_qrels = {q: qrels[q] & in_corpus for q in qids}
    conditions = {"base_informal": r_base, "adapted_informal": r_adapt, "base_formal": r_form}
    metrics = {name: per_query_metrics({q: [doc_ids[j] for j in rows[i]] for i, q in enumerate(qids)}, sample_qrels)
               for name, rows in conditions.items()}
    records = []
    for i, qid in enumerate(qids):
        relevant = qrels[qid] & in_corpus
        gold = torch.stack([retriever.doc_embedding(d) for d in sorted(relevant)])
        rec = {
            "query_id": qid, "formal_query": formal[i], "informal_query": informal[i],
            "cos_informal_formal": {"base": cos(z_inf[i], z_form[i]), "adapted": cos(z_adapt[i], z_form[i])},
            "cos_informal_best_gold": {"base": float((gold @ z_inf[i]).max()), "adapted": float((gold @ z_adapt[i]).max())},
            "cos_formal_best_gold": float((gold @ z_form[i]).max()),
            "cos_base_vs_adapted": cos(z_inf[i], z_adapt[i]),
            "first_relevant_rank": {"base_informal": first_relevant_rank(r_base[i], relevant, doc_ids),
                                    "adapted_informal": first_relevant_rank(r_adapt[i], relevant, doc_ids),
                                    "base_formal": first_relevant_rank(r_form[i], relevant, doc_ids)},
            "metrics": {name: {m: float(metrics[name][m][i]) for m in METRICS} for name in conditions},
            "top_adapted": [doc_ids[j] for j in r_adapt[i][: args.k]],
            "top_base": [doc_ids[j] for j in r_base[i][: args.k]],
        }
        records.append(rec)

        ranks = rec["first_relevant_rank"]
        print("\n" + "=" * 118)
        print(f"[{i + 1}/{len(qids)}] query_id {qid}  ({len(relevant)} relevant passage(s) in corpus)")
        print(f"  formal   : {formal[i]}")
        print(f"  informal : {informal[i]}")
        print(f"  {'':34} {'base':>8} {'adapted':>8} {'Δ':>8}")
        for label, key in [("cos(informal, formal query)", "cos_informal_formal"),
                           ("cos(informal, best gold passage)", "cos_informal_best_gold")]:
            b, a = rec[key]["base"], rec[key]["adapted"]
            print(f"  {label:<34} {b:>8.4f} {a:>8.4f} {a - b:>+8.4f}")
        print(f"  {'cos(formal, best gold passage)':<34} {rec['cos_formal_best_gold']:>8.4f}   (reference)")
        print(f"  {'cos(base informal, adapted)':<34} {rec['cos_base_vs_adapted']:>8.4f}   (how far the adapter moved it)")
        print(f"  first relevant rank: base informal {fmt_rank(ranks['base_informal'])} | "
              f"adapted informal {fmt_rank(ranks['adapted_informal'])} | base formal {fmt_rank(ranks['base_formal'])}")
        print_metrics_table(rec["metrics"])
        print_topk(f"top-{args.k} base BGE-M3 (informal):", s_base[i], r_base[i], retriever, args.k, relevant)
        print_topk(f"top-{args.k} Query-DANN (informal):", s_adapt[i], r_adapt[i], retriever, args.k, relevant)

    n = len(records)
    mean = lambda f: sum(f(r) for r in records) / n
    hit = lambda key: sum(1 for r in records if (r["first_relevant_rank"][key] or 10 ** 9) <= args.k)
    print("\n" + "=" * 118)
    print(f"SUMMARY over {n} sampled {args.split} queries")
    print(f"  mean cos(informal, formal)     base {mean(lambda r: r['cos_informal_formal']['base']):.4f} -> adapted {mean(lambda r: r['cos_informal_formal']['adapted']):.4f}")
    print(f"  mean cos(informal, best gold)  base {mean(lambda r: r['cos_informal_best_gold']['base']):.4f} -> adapted {mean(lambda r: r['cos_informal_best_gold']['adapted']):.4f}"
          f"   (formal {mean(lambda r: r['cos_formal_best_gold']):.4f})")
    print(f"  relevant passage in top-{args.k}   base informal {hit('base_informal')}/{n} | adapted informal {hit('adapted_informal')}/{n} | base formal {hit('base_formal')}/{n}")
    print_metrics_table({name: {m: float(metrics[name][m].mean()) for m in METRICS} for name in conditions},
                        title=f"mean retrieval metrics over {n} queries")
    return records


def free_text_demo(retriever: QueryDANNRetriever, args):
    z_base = retriever.encode(args.query)
    z_adapt = retriever.head.adapt(z_base)
    (s_base, r_base), (s_adapt, r_adapt) = retriever.search(z_base, args.k), retriever.search(z_adapt, args.k)
    records = []
    for i, q in enumerate(args.query):
        print("\n" + "=" * 118)
        print(f"query: {q}")
        print(f"  cos(base, adapted) = {cos(z_base[i], z_adapt[i]):.4f}")
        print_topk(f"top-{args.k} base BGE-M3:", s_base[i], r_base[i], retriever, args.k)
        print_topk(f"top-{args.k} Query-DANN:", s_adapt[i], r_adapt[i], retriever, args.k)
        ids = retriever.store.doc_ids
        records.append({"query": q, "cos_base_vs_adapted": cos(z_base[i], z_adapt[i]),
                        "top_base": [ids[j] for j in r_base[i]], "top_adapted": [ids[j] for j in r_adapt[i]]})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "v2_dann_g1.0", "best"))
    ap.add_argument("--query", action="append", help="Free-text query (repeatable); skips sampling")
    ap.add_argument("--n", type=int, default=5, help="Number of queries to sample")
    ap.add_argument("--qids", nargs="*", help="Specific query_ids instead of random sampling")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--k", type=int, default=5, help="Passages to display per query")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=None, help="Write per-query results JSON here")
    args = ap.parse_args()

    retriever = QueryDANNRetriever(args.ckpt, args.cache_dir, args.data_dir)
    records = free_text_demo(retriever, args) if args.query else sampled_demo(retriever, args)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
