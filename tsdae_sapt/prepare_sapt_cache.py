"""Embedding caches for the SAPT (TSDAE) model, in the same layout as ../cache.

  --mode query_only : SAPT encodes the queries; passages keep the ORIGINAL BGE-M3 embeddings
                      (corpus files are symlinked to ../cache, hard negatives copied). Shows how far the
                      query space drifted away from the frozen passage index.
  --mode symmetric  : SAPT encodes queries AND all 500k passages (~14 min, +1 GB); hard negatives re-mined.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "encoder_comparison"))

from dataset import load_corpus, load_qrels, load_train_pairs  # noqa: E402
from eval import dense_search  # noqa: E402
from encoders import EncoderSpec, GenericEncoder  # noqa: E402

BASE_CACHE = os.path.join(ROOT, "cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["query_only", "symmetric"])
    ap.add_argument("--model_path", default=os.path.join(HERE, "output", "bge-m3-sapt-informal-id"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--token_budget", type=int, default=48000)
    ap.add_argument("--mine_depth", type=int, default=200)
    ap.add_argument("--min_free_gb", type=float, default=2.0)
    args = ap.parse_args()

    cache_dir = os.path.join(HERE, "cache", f"sapt_{args.mode}")
    os.makedirs(cache_dir, exist_ok=True)
    path = lambda name: os.path.join(cache_dir, name)
    # Same pooling / prefixes / lengths as BGE-M3 (CLS, no prefixes), only the weights differ.
    spec = EncoderSpec(name=f"sapt_{args.mode}", label=f"BGE-M3 + SAPT ({args.mode})", model_id=args.model_path,
                       pooling="cls", query_prefix="", passage_prefix="", cache_dir=cache_dir, query_max_length=64)

    shutil.copyfile(os.path.join(BASE_CACHE, "splits.json"), path("splits.json"))
    tp = load_train_pairs(args.data_dir)
    qids = tp.query_id.tolist()
    with open(os.path.join(BASE_CACHE, "query_ids.json")) as f:
        assert json.load(f) == qids, "query order differs from ../cache"
    with open(path("query_ids.json"), "w") as f:
        json.dump(qids, f)

    corpus_done = os.path.exists(path("corpus_emb.npy")) and os.path.exists(path("corpus_ids.json"))
    if args.mode == "symmetric" and not corpus_done:
        free_gb = shutil.disk_usage(cache_dir).free / 1e9
        if free_gb < args.min_free_gb:
            raise SystemExit(f"only {free_gb:.1f} GB free; the symmetric corpus cache needs ~1 GB")

    enc = GenericEncoder(spec)
    np.save(path("formal_emb.npy"), enc.encode_queries(tp.formal_query.tolist(), desc="formal queries"))
    np.save(path("informal_emb.npy"), enc.encode_queries(tp.generated_informal_query.tolist(), desc="informal queries"))

    if args.mode == "query_only":
        for name in ("corpus_emb.npy", "corpus_ids.json"):
            if os.path.lexists(path(name)):
                os.remove(path(name))
            os.symlink(os.path.join(BASE_CACHE, name), path(name))
        shutil.copyfile(os.path.join(BASE_CACHE, "hard_negs.json"), path("hard_negs.json"))
        print(f"done: queries re-encoded; passages = original BGE-M3 index (symlinked) -> {cache_dir}")
        return

    doc_ids, passages = load_corpus(args.data_dir)
    if corpus_done:
        print("corpus_emb.npy exists, skipping corpus encoding")
    else:
        emb = np.lib.format.open_memmap(path("corpus_emb.npy.tmp"), mode="w+", dtype=np.float16, shape=(len(passages), spec.dim))
        enc.encode_passages(passages, token_budget=args.token_budget, out=emb, desc="corpus")
        emb.flush()
        del emb
        os.replace(path("corpus_emb.npy.tmp"), path("corpus_emb.npy"))
        with open(path("corpus_ids.json"), "w") as f:
            json.dump(doc_ids, f)
    del enc, passages
    torch.cuda.empty_cache()

    corpus = torch.from_numpy(np.load(path("corpus_emb.npy"))).cuda()
    formal = torch.from_numpy(np.load(path("formal_emb.npy"))).cuda()
    _, idx = dense_search(formal, corpus, k=args.mine_depth)
    qrels = load_qrels(args.data_dir)
    hard_negs = {q: [doc_ids[j] for j in row if doc_ids[j] not in qrels.get(q, set())] for q, row in zip(qids, idx.tolist())}
    with open(path("hard_negs.json"), "w") as f:
        json.dump(hard_negs, f)
    print(f"done: queries + {corpus.shape[0]:,} passages re-encoded -> {cache_dir} | "
          f"free disk {shutil.disk_usage(cache_dir).free / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
