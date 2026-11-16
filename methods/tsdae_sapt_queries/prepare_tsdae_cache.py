"""Build an embedding cache whose QUERY vectors come from a TSDAE-SAPT query tower.

The passage side stays the original frozen BGE-M3 (corpus_emb.npy / corpus_ids.json are symlinked from
../cache), so the index never changes and every comparison stays paired query-for-query. With --encoder me5 the
queries use mE5's pooling / prefix and the passages are mE5's own frozen index (methods/encoders.py cache_dir):

  <out_cache>/corpus_emb.npy   -> ../cache/corpus_emb.npy      (symlink, frozen document tower)
  <out_cache>/corpus_ids.json  -> ../cache/corpus_ids.json     (symlink)
  <out_cache>/splits.json, hard_negs.json                       (copied: identical splits/negatives)
  <out_cache>/query_ids.json, formal_emb.npy, informal_emb.npy  (re-encoded with the DAPT'd tower)

  python3 methods/tsdae_sapt_queries/prepare_tsdae_cache.py --model methods/tsdae_sapt_queries/outputs/tsdae_informal/model
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from preprocessing.dataset import load_train_pairs  # noqa: E402
from methods.encoders import ENCODERS, GenericEncoder  # noqa: E402
from methods.model import BGEM3Encoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Directory of the TSDAE-adapted query encoder")
    ap.add_argument("--out_cache", default=None, help="Default: <model>/../cache")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--query_max_length", type=int, default=None, help="Default: the encoder's (bge_m3: 64)")
    ap.add_argument("--encoder", default="bge_m3", choices=list(ENCODERS),
                    help="Encoder mode (methods/encoders.py): query pooling / prefix and which frozen index to link")
    ap.add_argument("--base_cache", default=None,
                    help="Base cache to mirror (default: the encoder's; data/official: methods/cache/official)")
    args = ap.parse_args()
    spec = ENCODERS[args.encoder]
    reference = args.base_cache or spec.cache_dir  # bge_m3: methods/cache
    max_length = args.query_max_length or spec.query_max_length

    out = args.out_cache or os.path.join(os.path.dirname(os.path.abspath(args.model)), "cache")
    os.makedirs(out, exist_ok=True)
    path = lambda name: os.path.join(out, name)

    # Frozen document side: symlink, never re-encoded.
    for name in ("corpus_emb.npy", "corpus_ids.json"):
        link, target = path(name), os.path.join(reference, name)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.relpath(target, out), link)
    for name in ("splits.json", "hard_negs.json"):
        shutil.copyfile(os.path.join(reference, name), path(name))

    tp = load_train_pairs(args.data_dir)
    qids = tp.query_id.tolist()
    with open(os.path.join(reference, "query_ids.json")) as f:
        assert json.load(f) == qids, "query order differs from ../cache"
    with open(path("query_ids.json"), "w") as f:
        json.dump(qids, f)

    # Pooling + L2 norm exactly as the frozen document tower was encoded (bge_m3: CLS, ../prepare.py).
    if args.encoder == "bge_m3":
        enc = BGEM3Encoder(args.model)
        encode = lambda texts, desc: enc.encode(texts, max_length=max_length, desc=desc)
    else:
        enc = GenericEncoder(replace(spec, model_id=args.model, cache_dir=out, query_max_length=max_length))
        encode = lambda texts, desc: enc.encode_queries(texts, desc=desc)
    columns = [("formal_query", "formal_emb.npy"), ("generated_informal_query", "informal_emb.npy")]
    if "informal_varied" in tp:  # data/official: the second informal style of the test queries
        columns.append(("informal_varied", "informal_varied_emb.npy"))
    for col, name in columns:
        np.save(path(name), encode(tp[col].tolist(), f"{name} ({os.path.basename(args.model)})"))

    base = np.load(os.path.join(reference, "informal_emb.npy")).astype("float32")
    new = np.load(path("informal_emb.npy")).astype("float32")
    cos = float((base * new).sum(1).mean())
    print(f"done: {out}\n  mean cos(base informal, TSDAE informal) = {cos:.4f}  (1.0 = unchanged tower)")


if __name__ == "__main__":
    main()
