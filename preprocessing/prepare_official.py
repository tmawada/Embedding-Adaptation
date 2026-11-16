"""data/official/: train on official MIRACL-id train, pick checkpoints on our dev split, test on official MIRACL-id dev.

  train  our old train + old test split (both from official MIRACL-id train)                     3,444
  dev    our old dev split (checkpoint selection), minus queries that repeat a train question   ~370
  test   the official MIRACL-id dev queries kept by comparison/prepare_testset.py (data/miracl_dev/), never used to
         train the encoders                                                                          867

Files, in the layout of data/ so every trainer runs on it with --data_dir data/official:
  train_pairs.csv   query_id, formal, informal, informal_varied. Test queries: informal = template style, informal_varied
                    = varied style (preprocessing/generate_dev_informal.py). Other rows have one informal version
                    (data/train_pairs.csv), repeated in informal_varied.
  qrels.json        data/qrels.json + data/miracl_dev/qrels.json
  splits.json       the split above, used as is by methods/prepare.py --splits_file; *_excluded lists say what was left out
  miracl_pairs.json, corpus.json   symlinks to ../ (training triples and passages are unchanged)

Same question = same formal or informal text once case, spacing and punctuation are ignored
(comparison/prepare_testset.py norm).

  python3 preprocessing/prepare_official.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "comparison"))

from preprocessing.dataset import load_train_pairs  # noqa: E402
from prepare_testset import DEV_DATA, norm  # noqa: E402

OUT = os.path.join(ROOT, "data", "official")


def read_jsonl(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return {r["query_id"]: r for r in map(json.loads, filter(str.strip, f))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--old_splits", default=os.path.join(ROOT, "methods", "cache", "splits.json"))
    args = ap.parse_args()

    tp = load_train_pairs(args.data_dir).set_index("query_id")
    with open(args.old_splits) as f:
        old = json.load(f)
    with open(os.path.join(DEV_DATA, "queries.json")) as f:
        test = [(r["query_id"], r["query"]) for r in json.load(f)]
    styles = {s: read_jsonl(os.path.join(DEV_DATA, f"informal_{s}.jsonl")) for s in ("template", "varied")}
    missing = {s: [q for q, _ in test if q not in got] for s, got in styles.items()}
    if any(missing.values()):
        raise SystemExit("informal versions missing, run preprocessing/generate_dev_informal.py first: "
                         + ", ".join(f"{s} {len(m)}" for s, m in missing.items()))

    train = sorted(old["train"] + old["test"], key=int)
    texts = lambda q: {norm(tp.at[q, "formal_query"]), norm(tp.at[q, "generated_informal_query"])}
    seen = set().union(*(texts(q) for q in train))
    dev = [q for q in old["dev"] if not texts(q) & seen]
    dev_repeats = sorted(set(old["dev"]) - set(dev), key=int)
    splits = {"train": train, "dev": dev, "test": [q for q, _ in test],
              "twins_excluded": old["twins_excluded"], "dev_repeats_train": dev_repeats,
              "unanswerable_excluded": old["unanswerable_excluded"]}

    rows = [{"query_id": q, "formal": tp.at[q, "formal_query"], "informal": tp.at[q, "generated_informal_query"],
             "informal_varied": tp.at[q, "generated_informal_query"]} for q in tp.index]
    rows += [{"query_id": q, "formal": text, "informal": styles["template"][q]["informal"],
              "informal_varied": styles["varied"][q]["informal"]} for q, text in test]
    with open(os.path.join(args.data_dir, "qrels.json")) as f:
        qrels = json.load(f)
    with open(os.path.join(DEV_DATA, "qrels.json")) as f:
        qrels += json.load(f)

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "train_pairs.csv"), index=False)
    for name, obj in (("qrels.json", qrels), ("splits.json", splits)):
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(obj, f)
    for name in ("miracl_pairs.json", "corpus.json"):
        link = os.path.join(OUT, name)
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(os.path.join(args.data_dir, name), OUT), link)
    models = {s: sorted({r["model"] for r in got.values()}) for s, got in styles.items()}
    print(f"data/official: train {len(train)} | dev {len(dev)} ({len(dev_repeats)} old dev queries repeat a train "
          f"question, left out) | test {len(test)} (informal: template + varied, {models}) | "
          f"{len(rows)} query rows, {len(qrels)} relevance labels")


if __name__ == "__main__":
    main()
