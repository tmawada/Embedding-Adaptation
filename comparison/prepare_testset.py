"""Extra test sets for the free checks: no API calls, no retraining (comparison/evaluate_all.py --testset scores them).

  style_pilot        the 83 test queries that also have a style-sampled informal rewrite (../BGE_M3/data/pairs/
                     new_pairs.csv, made by ../BGE_M3/preprocess/generate_informal.py). Same questions, two informal
                     styles: the training template ("... apa sih") and the varied style.
  miracl_dev_formal  the official MIRACL-id dev queries (data/topics.miracl-v1.0-id-dev.tsv), formal side only. Unlike
                     our queries (all from official MIRACL-id train), the encoders were never trained on them.

Per test set and encoder, methods/cache/testsets/<testset>/<encoder>/ has the layout dataset.EmbeddingStore reads:
  query_ids.json, splits.json ({"test": ids}), formal_emb.npy, informal_emb.npy (style_pilot: the template style),
  informal_varied_emb.npy (style_pilot), corpus_emb.npy / corpus_ids.json (symlinks: the encoder's frozen index)
  towers/<method>__<run>/   the same query files from a fine-tuned query encoder (InfoNCE, TSDAE final), for every
                            run whose weights still exist (run_all.py prunes them after the seed runs)
style_pilot copies the formal and template vectors from the existing caches (the exact vectors behind the reported
numbers) and only encodes the varied style. miracl_dev_formal also writes data/miracl_dev/{queries,qrels,excluded}.json:
the dev queries with a relevant passage in the 500k index, minus those whose text repeats an official-train query or
an earlier dev query (same text once case, spacing and punctuation are ignored). Every encoder is first checked against the
cache it built: re-encoding our test queries must reproduce the cached vectors.

  python3 comparison/prepare_testset.py                                                            # both, BGE-M3
  python3 comparison/prepare_testset.py --testset miracl_dev_formal --encoder mpnet --towers false   # yardstick
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from preprocessing.dataset import load_train_pairs  # noqa: E402
from methods.encoders import ENCODERS, GenericEncoder  # noqa: E402
from methods.model import BGEM3Encoder  # noqa: E402
from run_all import SEEDS, artifacts  # noqa: E402

TESTSETS = ("style_pilot", "miracl_dev_formal")
DEV_DATA = os.path.join(ROOT, "data", "miracl_dev")
STYLE_PAIRS = os.path.join(os.path.dirname(ROOT), "BGE_M3", "data", "pairs", "new_pairs.csv")
DEV_QRELS = os.path.join(os.path.dirname(ROOT), "BGE_M3", "data", "qrels", "qrels_dev.tsv")
# Fine-tuned query encoders: (artifact key of the weights, artifact key of their query cache over our queries).
TOWERS = (("query_infonce_model", "query_infonce_cache"),
          ("tsdae_sapt_queries_final", "tsdae_sapt_queries_final_cache"))
MIN_COS = 0.999  # re-encoded vs cached query vectors (fp16, different batches)

# Same question = same text after lower-casing and ignoring punctuation and spacing. Stricter than
# preprocessing.dataset.drop_twins (case and spacing only), which misses pairs such as "Eropa ?" / "Eropa?".
norm = lambda s: " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def testset_dir(testset: str, enc: str = None) -> str:
    """methods/cache/testsets/<testset>[/<encoder>]: the encoder-independent query texts live one level up."""
    d = os.path.join(ROOT, "methods", "cache", "testsets", testset)
    return d if enc is None else os.path.join(d, enc)


def tower_key(cache: str) -> str:
    """methods/<method>/outputs/<run>/<cache> -> <method>__<run>: one directory per fine-tuned query encoder."""
    parts = os.path.normpath(cache).split(os.sep)
    return f"{parts[-4]}__{parts[-2]}"


def tower_dir(testset: str, enc: str, cache: str) -> str:
    return os.path.join(testset_dir(testset, enc), "towers", tower_key(cache))


def has_weights(model_dir: str) -> bool:
    return os.path.isdir(model_dir) and any(n.endswith((".safetensors", ".bin")) for n in os.listdir(model_dir))


def dump(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------- #
# Queries and relevance
# --------------------------------------------------------------------------- #
def read_topics(path: str) -> dict:
    """Official MIRACL topics file: query_id <TAB> query, no header."""
    with open(path, encoding="utf-8", newline="") as f:
        return {row[0]: row[1] for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}


def build_miracl_dev(data_dir: str, qrels_path: str, corpus_ids: list) -> list:
    """data/miracl_dev/{queries,qrels,excluded}.json in the layout of data/queries.json / qrels.json; returns kept ids."""
    topics = read_topics(os.path.join(data_dir, "topics.miracl-v1.0-id-dev.tsv"))
    qrels = {}
    with open(qrels_path) as f:
        for line in f:  # query_id Q0 doc_id relevance; data/qrels.json keeps the positives only
            q, _, doc, rel = line.split()
            if int(rel) > 0:
                qrels.setdefault(q, []).append(doc)
    in_corpus = set(corpus_ids)
    with open(os.path.join(data_dir, "queries.json")) as f:
        train = {norm(r["query"]) for r in json.load(f)}  # data/queries.json = every official-train query
    reasons = ("no relevant passage in the 500k index", "same text as an official-train query",
               "repeats an earlier dev query")
    excluded, seen, kept = {r: [] for r in reasons}, set(), []
    for q in sorted(topics, key=int):
        text = norm(topics[q])
        if not any(d in in_corpus for d in qrels.get(q, [])):
            excluded[reasons[0]].append(q)
        elif text in train:
            excluded[reasons[1]].append(q)
        elif text in seen:
            excluded[reasons[2]].append(q)
        else:
            seen.add(text)
            kept.append(q)
    os.makedirs(DEV_DATA, exist_ok=True)
    dump([{"query_id": q, "query": topics[q]} for q in kept], os.path.join(DEV_DATA, "queries.json"))
    dump([{"query_id": q, "relevant_doc": d} for q in kept for d in qrels[q]], os.path.join(DEV_DATA, "qrels.json"))
    dump({"topics": os.path.relpath(os.path.join(data_dir, "topics.miracl-v1.0-id-dev.tsv"), ROOT),
          "qrels": os.path.relpath(qrels_path, ROOT), "n_topics": len(topics), "n_kept": len(kept),
          "excluded": excluded}, os.path.join(DEV_DATA, "excluded.json"))
    print(f"miracl_dev: {len(topics)} topics -> {len(kept)} kept; "
          + ", ".join(f"{len(v)} {k}" for k, v in excluded.items()))
    return kept


def build_style_pilot(data_dir: str, pairs_path: str, test_ids: list) -> list:
    """Our test queries that new_pairs.csv also rewrote: formal text, template informal, varied informal, style."""
    pairs = pd.read_csv(pairs_path, dtype=str).set_index("query_id")
    tp = load_train_pairs(data_dir).set_index("query_id")
    ids = sorted(set(pairs.index) & set(test_ids), key=int)
    differ = [q for q in ids if pairs.at[q, "formal"] != tp.at[q, "formal_query"]]
    assert not differ, f"formal text in {pairs_path} differs from train_pairs.csv for {differ}"
    rows = [{"query_id": q, "formal": tp.at[q, "formal_query"], "informal_template": tp.at[q, "generated_informal_query"],
             "informal_varied": pairs.at[q, "informal"], "style": pairs.at[q, "style"]} for q in ids]
    os.makedirs(testset_dir("style_pilot"), exist_ok=True)
    dump(rows, os.path.join(testset_dir("style_pilot"), "queries.json"))
    print(f"style_pilot: {len(ids)} of the {len(test_ids)} test queries have a varied-style rewrite "
          "(formal text identical in both files)")
    return rows


# --------------------------------------------------------------------------- #
# Query encoders, exactly as the existing caches were built
# --------------------------------------------------------------------------- #
def base_encoder(enc: str):
    """methods/prepare.py (bge_m3) / methods/prepare_encoder.py (the others)."""
    spec = ENCODERS[enc]
    if enc == "bge_m3":
        model = BGEM3Encoder(spec.model_id)
        return lambda texts: model.encode(texts, max_length=spec.query_max_length, progress=False)
    model = GenericEncoder(spec)
    return lambda texts: model.encode_queries(texts, progress=False)


def tower_encoder(enc: str, weights_key: str, model_dir: str):
    """methods/query_infonce/prepare_infonce_cache.py / methods/tsdae_sapt_queries/prepare_tsdae_cache.py."""
    spec = ENCODERS[enc]
    if enc == "bge_m3" and weights_key == "tsdae_sapt_queries_final":
        model = BGEM3Encoder(model_dir)
        return lambda texts: model.encode(texts, max_length=spec.query_max_length, progress=False)
    model = GenericEncoder(replace(spec, model_id=model_dir))
    return lambda texts: model.encode_queries(texts, progress=False)


def cached_rows(cache_dir: str, qids: list, name: str) -> np.ndarray:
    with open(os.path.join(cache_dir, "query_ids.json")) as f:
        row = {q: i for i, q in enumerate(json.load(f))}
    return np.load(os.path.join(cache_dir, name), mmap_mode="r")[[row[q] for q in qids]]


def check_encoder(encode, cache_dir: str, data_dir: str, n: int = 256) -> float:
    """Min cosine between re-encoded and cached formal vectors of our first n test queries: the new test-set vectors
    are only comparable with the reported numbers if the encoding path reproduces the cache."""
    with open(os.path.join(cache_dir, "splits.json")) as f:
        qids = json.load(f)["test"][:n]
    tp = load_train_pairs(data_dir).set_index("query_id")
    new = encode([tp.at[q, "formal_query"] for q in qids]).astype(np.float32)
    return float((cached_rows(cache_dir, qids, "formal_emb.npy").astype(np.float32) * new).sum(1).min())


def write_cache(out: str, qids: list, vectors: dict, index_dir: str = None):
    """Query files (+ symlinked passage index for a base cache) in the EmbeddingStore layout."""
    os.makedirs(out, exist_ok=True)
    dump(qids, os.path.join(out, "query_ids.json"))
    dump({"test": qids}, os.path.join(out, "splits.json"))
    for name, v in vectors.items():
        np.save(os.path.join(out, name), np.asarray(v, dtype=np.float16))
    for name in ("corpus_emb.npy", "corpus_ids.json") if index_dir else ():
        link = os.path.join(out, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.relpath(os.path.join(index_dir, name), out), link)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="all", choices=[*TESTSETS, "all"])
    ap.add_argument("--encoder", default="bge_m3", choices=list(ENCODERS))
    ap.add_argument("--towers", default="true", choices=["true", "false"],
                    help="Also encode with every fine-tuned query encoder (InfoNCE, TSDAE final) whose weights exist")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--style_pairs", default=STYLE_PAIRS, help="query_id, formal, informal, style (new_pairs.csv)")
    ap.add_argument("--dev_qrels", default=DEV_QRELS, help="Official MIRACL-id dev qrels (query Q0 doc relevance)")
    args = ap.parse_args()
    spec = ENCODERS[args.encoder]
    with open(os.path.join(spec.cache_dir, "splits.json")) as f:
        test_ids = json.load(f)["test"]
    with open(os.path.join(spec.cache_dir, "corpus_ids.json")) as f:
        corpus_ids = json.load(f)

    # testset -> (query ids, {file: texts to encode}, files copied from the existing cache)
    work = {}
    if args.testset in ("style_pilot", "all"):
        rows = build_style_pilot(args.data_dir, args.style_pairs, test_ids)
        work["style_pilot"] = ([r["query_id"] for r in rows],
                               {"informal_varied_emb.npy": [r["informal_varied"] for r in rows]},
                               ("formal_emb.npy", "informal_emb.npy"))
    if args.testset in ("miracl_dev_formal", "all"):
        kept = build_miracl_dev(args.data_dir, args.dev_qrels, corpus_ids)
        with open(os.path.join(DEV_DATA, "queries.json")) as f:
            text = {r["query_id"]: r["query"] for r in json.load(f)}
        work["miracl_dev_formal"] = (kept, {"formal_emb.npy": [text[q] for q in kept]}, ())

    # (name, cache it reproduces, output dir per test set, encoder factory, passage index to link)
    jobs = [("base", spec.cache_dir, lambda ts: testset_dir(ts, args.encoder),
             lambda: base_encoder(args.encoder), spec.cache_dir)]
    if args.towers == "true":
        for seed in SEEDS:
            A = artifacts(args.encoder, seed)
            for wkey, ckey in TOWERS:
                model_dir, cache = os.path.join(ROOT, A[wkey]), os.path.join(ROOT, A[ckey])
                if has_weights(model_dir) and os.path.exists(os.path.join(cache, "formal_emb.npy")):
                    jobs.append((tower_key(A[ckey]), cache, lambda ts, c=A[ckey]: tower_dir(ts, args.encoder, c),
                                 lambda w=wkey, m=model_dir: tower_encoder(args.encoder, w, m), None))
    print(f"[{args.encoder}] encoders: {', '.join(j[0] for j in jobs)}")

    for name, cache, out_dir, factory, index_dir in jobs:
        encode = factory()
        cos = check_encoder(encode, cache, args.data_dir)
        assert cos >= MIN_COS, f"{name}: re-encoded test queries differ from {cache} (min cos {cos:.5f})"
        for ts, (qids, to_encode, to_copy) in work.items():
            vectors = {f: cached_rows(cache, qids, f) for f in to_copy}
            vectors.update({f: encode(texts) for f, texts in to_encode.items()})
            write_cache(out_dir(ts), qids, vectors, index_dir)
        print(f"  {name}: min cos vs its cache {cos:.5f}; wrote {', '.join(work)}")
        del encode
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
