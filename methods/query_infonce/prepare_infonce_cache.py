"""Embedding cache for the InfoNCE-tuned encoder, query-side only, in the same layout as ../cache.

  The passage index is SYMLINKED to ../cache and never re-encoded: the whole point of this method is
  that the document space stays frozen at base BGE-M3 while the query tower moves toward it. There is
  deliberately no --mode symmetric (unlike ../tsdae_sapt), because re-encoding passages would both
  contradict that constraint and cost ~1 GB this machine does not have.

  query_max_length stays 64 (methods/encoders.py ENCODERS["bge_m3"]), not the 128 used in
  training, so the ONLY difference between the base and InfoNCE conditions is the weights — no
  truncation confound in the comparison. --encoder me5 takes mE5's pooling / prefix / length the same way
  and links mE5's own frozen index.

  python3 models/query_infonce/prepare_infonce_cache.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from preprocessing.dataset import load_train_pairs  # noqa: E402
from methods.encoders import ENCODERS, GenericEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=os.path.join(HERE, "outputs", "bge-m3-infonce-id"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out_cache", default=os.path.join(HERE, "cache", "infc_query_only"))
    ap.add_argument("--encoder", default="bge_m3", choices=list(ENCODERS),
                    help="Encoder mode (methods/encoders.py): query pooling / prefix / length and which index to link")
    ap.add_argument("--base_cache", default=None,
                    help="Base cache to mirror (default: the encoder's; data/official: methods/cache/official)")
    args = ap.parse_args()
    base = ENCODERS[args.encoder]
    base_cache = os.path.abspath(args.base_cache or base.cache_dir)  # bge_m3: methods/cache; absolute for symlinks

    cache_dir = args.out_cache
    os.makedirs(cache_dir, exist_ok=True)
    path = lambda name: os.path.join(cache_dir, name)
    # Same pooling / prefixes / lengths as the base encoder (bge_m3: CLS, no prefixes); only the weights differ.
    spec = replace(base, name="infc_query_only", label=f"{base.label} + InfoNCE (query-only)",
                   model_id=args.model_path, cache_dir=cache_dir)

    shutil.copyfile(os.path.join(base_cache, "splits.json"), path("splits.json"))
    tp = load_train_pairs(args.data_dir)
    qids = tp.query_id.tolist()
    with open(os.path.join(base_cache, "query_ids.json")) as f:
        assert json.load(f) == qids, "query order differs from ../cache"
    with open(path("query_ids.json"), "w") as f:
        json.dump(qids, f)

    enc = GenericEncoder(spec)
    np.save(path("formal_emb.npy"), enc.encode_queries(tp.formal_query.tolist(), desc="formal queries"))
    np.save(path("informal_emb.npy"), enc.encode_queries(tp.generated_informal_query.tolist(),
                                                         desc="informal queries"))
    if "informal_varied" in tp:  # data/official: the second informal style of the test queries
        np.save(path("informal_varied_emb.npy"), enc.encode_queries(tp.informal_varied.tolist(),
                                                                    desc="informal queries, varied style"))

    # Passages = the base encoder's original index, symlinked.
    for name in ("corpus_emb.npy", "corpus_ids.json"):
        if os.path.lexists(path(name)):
            os.remove(path(name))
        os.symlink(os.path.join(base_cache, name), path(name))
    shutil.copyfile(os.path.join(base_cache, "hard_negs.json"), path("hard_negs.json"))
    print(f"done: queries re-encoded; passages = original {base.label} index (symlinked) -> {cache_dir}")


if __name__ == "__main__":
    main()
