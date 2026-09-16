"""Build an encoder-specific embedding cache with the same layout as ../cache (cf. ../prepare.py).

  python3 encoder_comparison/prepare_encoder.py --encoder me5_large_instruct

Writes <spec.cache_dir>/: splits.json (copied: identical queries), query_ids.json, formal_emb.npy,
informal_emb.npy, corpus_ids.json, corpus_emb.npy, hard_negs.json (mined with this encoder's own ranking).
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

from dataset import load_corpus, load_qrels, load_train_pairs  # noqa: E402
from eval import dense_search  # noqa: E402
from encoders import ENCODERS, GenericEncoder  # noqa: E402

REFERENCE_CACHE = os.path.join(ROOT, "cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=[n for n in ENCODERS if n != "bge_m3"],
                    help="bge_m3's cache is ../cache, built by ../prepare.py")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--token_budget", type=int, default=48000)
    ap.add_argument("--mine_depth", type=int, default=200)
    ap.add_argument("--min_free_gb", type=float, default=2.5, help="Refuse to start encoding with less free disk")
    args = ap.parse_args()
    spec = ENCODERS[args.encoder]
    os.makedirs(spec.cache_dir, exist_ok=True)
    path = lambda name: os.path.join(spec.cache_dir, name)
    corpus_done = os.path.exists(path("corpus_emb.npy")) and os.path.exists(path("corpus_ids.json"))

    free_gb = shutil.disk_usage(spec.cache_dir).free / 1e9
    print(f"[{spec.name}] {spec.model_id} -> {spec.cache_dir} | free disk {free_gb:.1f} GB")
    if not corpus_done and free_gb < args.min_free_gb:
        raise SystemExit(f"only {free_gb:.1f} GB free; need {args.min_free_gb} GB for model weights + corpus cache")

    # Identical splits and row orders as the BGE-M3 cache, so results are paired query-for-query.
    shutil.copyfile(os.path.join(REFERENCE_CACHE, "splits.json"), path("splits.json"))
    tp = load_train_pairs(args.data_dir)
    qids = tp.query_id.tolist()
    with open(os.path.join(REFERENCE_CACHE, "query_ids.json")) as f:
        assert json.load(f) == qids, "query order differs from ../cache"
    with open(path("query_ids.json"), "w") as f:
        json.dump(qids, f)
    doc_ids, passages = load_corpus(args.data_dir)

    enc = GenericEncoder(spec)
    np.save(path("formal_emb.npy"), enc.encode_queries(tp.formal_query.tolist(), desc="formal queries"))
    np.save(path("informal_emb.npy"), enc.encode_queries(tp.generated_informal_query.tolist(), desc="informal queries"))

    if corpus_done:
        print("corpus_emb.npy exists, skipping corpus encoding")
    else:
        emb = np.lib.format.open_memmap(path("corpus_emb.npy.tmp"), mode="w+", dtype=np.float16,
                                        shape=(len(passages), spec.dim))
        enc.encode_passages(passages, token_budget=args.token_budget, out=emb, desc="corpus")
        emb.flush()
        del emb
        os.replace(path("corpus_emb.npy.tmp"), path("corpus_emb.npy"))
        with open(path("corpus_ids.json"), "w") as f:
            json.dump(doc_ids, f)
    del enc, passages
    torch.cuda.empty_cache()

    # Hard negatives from this encoder's ranking of the formal queries (same rule as ../prepare.py).
    corpus = torch.from_numpy(np.load(path("corpus_emb.npy"))).cuda()
    formal = torch.from_numpy(np.load(path("formal_emb.npy"))).cuda()
    _, idx = dense_search(formal, corpus, k=args.mine_depth)
    qrels = load_qrels(args.data_dir)
    hard_negs = {q: [doc_ids[j] for j in row if doc_ids[j] not in qrels.get(q, set())] for q, row in zip(qids, idx.tolist())}
    with open(path("hard_negs.json"), "w") as f:
        json.dump(hard_negs, f)
    print(f"done: corpus {tuple(corpus.shape)}, queries {tuple(formal.shape)}, "
          f"free disk {shutil.disk_usage(spec.cache_dir).free / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
