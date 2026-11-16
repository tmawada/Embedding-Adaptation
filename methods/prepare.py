"""One-off preprocessing with the frozen BGE-M3 encoder.

Because the encoder is frozen and the adapter sits on top of its pooled output,
encoding every text once is mathematically identical to encoding it inside the
training loop, and makes training/evaluation orders of magnitude faster.

Produces in --cache_dir:
  splits.json        query-level train/dev/test split (dev / test without repeated questions)
  query_ids.json     row order of the query embedding matrices
  formal_emb.npy     (n_queries, 1024) float16
  informal_emb.npy   (n_queries, 1024) float16
  corpus_ids.json    row order of corpus_emb.npy
  corpus_emb.npy     (n_passages, 1024) float16
  hard_negs.json     {query_id: [doc_id, ...]} base-model ranking on the formal
                     query with all labelled positives removed
  informal_varied_emb.npy  only when train_pairs.csv has an informal_varied column (data/official: a second informal
                     style of the test queries)

The official layout (data/official, preprocessing/prepare_official.py) passes --splits_file (its fixed split, used
as is) and --corpus_from methods/cache (the passage index is linked, not encoded again).
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

from preprocessing.dataset import drop_twins, load_corpus, load_miracl_pairs, load_qrels, load_train_pairs, make_splits
from methods.eval import dense_search
from methods.model import BGEM3Encoder, EMB_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--doc_max_length", type=int, default=512)
    ap.add_argument("--query_max_length", type=int, default=64)
    ap.add_argument("--token_budget", type=int, default=48000)
    ap.add_argument("--mine_depth", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop_twins", default="true", choices=["true", "false"],
                    help="Remove dev / test queries that repeat a train (or earlier dev / test) question")
    ap.add_argument("--splits_file", default=None, help="Use this splits.json as is instead of drawing a split")
    ap.add_argument("--corpus_from", default=None,
                    help="Cache dir whose corpus_emb.npy / corpus_ids.json are symlinked instead of encoding the corpus")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    path = lambda name: os.path.join(args.cache_dir, name)
    for name in ("corpus_emb.npy", "corpus_ids.json") if args.corpus_from else ():
        if not os.path.lexists(path(name)):
            os.symlink(os.path.relpath(os.path.join(args.corpus_from, name), args.cache_dir), path(name))

    tp = load_train_pairs(args.data_dir)
    qids = tp.query_id.tolist()
    doc_ids, passages = load_corpus(args.data_dir)
    in_corpus = set(doc_ids)

    # Only queries with at least one relevant passage in this corpus can be trained on or evaluated.
    answerable = sorted({p["query_id"] for p in load_miracl_pairs(args.data_dir) if p["positive_doc_id"] in in_corpus}, key=int)
    if args.splits_file:
        with open(args.splits_file) as f:
            splits = json.load(f)
    else:
        splits = make_splits(answerable, seed=args.seed)
        if args.drop_twins == "true":
            splits = drop_twins(splits, args.data_dir)
        splits["unanswerable_excluded"] = sorted(set(qids) - set(answerable), key=int)
    with open(path("splits.json"), "w") as f:
        json.dump(splits, f)
    print({k: len(v) for k, v in splits.items()})

    enc = BGEM3Encoder(args.model)

    with open(path("query_ids.json"), "w") as f:
        json.dump(qids, f)
    columns = [("formal_query", "formal_emb.npy"), ("generated_informal_query", "informal_emb.npy")]
    if "informal_varied" in tp:
        columns.append(("informal_varied", "informal_varied_emb.npy"))
    for col, name in columns:
        emb = enc.encode(tp[col].tolist(), max_length=args.query_max_length, desc=col)
        np.save(path(name), emb)

    if os.path.exists(path("corpus_emb.npy")) and os.path.exists(path("corpus_ids.json")):
        print("corpus_emb.npy exists, skipping corpus encoding")
    else:
        emb = np.lib.format.open_memmap(path("corpus_emb.npy.tmp"), mode="w+", dtype=np.float16, shape=(len(passages), EMB_DIM))
        enc.encode(passages, max_length=args.doc_max_length, token_budget=args.token_budget, out=emb, desc="corpus")
        emb.flush()
        del emb
        os.replace(path("corpus_emb.npy.tmp"), path("corpus_emb.npy"))
        with open(path("corpus_ids.json"), "w") as f:
            json.dump(doc_ids, f)
    del enc, passages
    torch.cuda.empty_cache()

    # Hard-negative mining with the base model on formal queries.
    corpus = torch.from_numpy(np.load(path("corpus_emb.npy"))).cuda()
    formal = torch.from_numpy(np.load(path("formal_emb.npy"))).cuda()
    _, idx = dense_search(formal, corpus, k=args.mine_depth)
    qrels = load_qrels(args.data_dir)
    hard_negs = {q: [doc_ids[j] for j in row if doc_ids[j] not in qrels.get(q, set())] for q, row in zip(qids, idx.tolist())}
    with open(path("hard_negs.json"), "w") as f:
        json.dump(hard_negs, f)
    print("done:", sorted(os.listdir(args.cache_dir)))


if __name__ == "__main__":
    main()
