"""Build an embedding cache whose QUERY vectors come from a TSDAE-SAPT query tower.

The passage side stays the original frozen BGE-M3 (corpus_emb.npy / corpus_ids.json are symlinked from
../cache), so the index never changes and every comparison stays paired query-for-query:

  <out_cache>/corpus_emb.npy   -> ../../cache/corpus_emb.npy   (symlink, frozen document tower)
  <out_cache>/corpus_ids.json  -> ../../cache/corpus_ids.json  (symlink)
  <out_cache>/splits.json, hard_negs.json                       (copied: identical splits/negatives)
  <out_cache>/query_ids.json, formal_emb.npy, informal_emb.npy  (re-encoded with the DAPT'd tower)

  python3 tsdae_sapt_queries/prepare_tsdae_cache.py --model tsdae_sapt_queries/runs/tsdae_informal/model
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset import load_train_pairs  # noqa: E402
from model import BGEM3Encoder  # noqa: E402

REFERENCE_CACHE = os.path.join(ROOT, "cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Directory of the TSDAE-adapted query encoder")
    ap.add_argument("--out_cache", default=None, help="Default: <model>/../cache")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--query_max_length", type=int, default=64)
    args = ap.parse_args()

    out = args.out_cache or os.path.join(os.path.dirname(os.path.abspath(args.model)), "cache")
    os.makedirs(out, exist_ok=True)
    path = lambda name: os.path.join(out, name)

    # Frozen document side: symlink, never re-encoded.
    for name in ("corpus_emb.npy", "corpus_ids.json"):
        link, target = path(name), os.path.join(REFERENCE_CACHE, name)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.relpath(target, out), link)
    for name in ("splits.json", "hard_negs.json"):
        shutil.copyfile(os.path.join(REFERENCE_CACHE, name), path(name))

    tp = load_train_pairs(args.data_dir)
    qids = tp.query_id.tolist()
    with open(os.path.join(REFERENCE_CACHE, "query_ids.json")) as f:
        assert json.load(f) == qids, "query order differs from ../cache"
    with open(path("query_ids.json"), "w") as f:
        json.dump(qids, f)

    # CLS pooling + L2 norm, exactly as the frozen document tower was encoded (../prepare.py).
    enc = BGEM3Encoder(args.model)
    for col, name in [("formal_query", "formal_emb.npy"), ("generated_informal_query", "informal_emb.npy")]:
        emb = enc.encode(tp[col].tolist(), max_length=args.query_max_length, desc=f"{name} ({os.path.basename(args.model)})")
        np.save(path(name), emb)

    base = np.load(os.path.join(REFERENCE_CACHE, "informal_emb.npy")).astype("float32")
    new = np.load(path("informal_emb.npy")).astype("float32")
    cos = float((base * new).sum(1).mean())
    print(f"done: {out}\n  mean cos(base informal, TSDAE informal) = {cos:.4f}  (1.0 = unchanged tower)")


if __name__ == "__main__":
    main()
