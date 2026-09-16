"""Data pipeline: raw MIRACL-id files -> splits -> cached-embedding training batches."""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = "data"
CACHE_DIR = "cache"


# --------------------------------------------------------------------------- #
# Raw loaders
# --------------------------------------------------------------------------- #
def load_train_pairs(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """train_pairs.csv -> [query_id, formal_query, generated_informal_query]."""
    df = pd.read_csv(os.path.join(data_dir, "train_pairs.csv"), dtype={"query_id": str})
    return df.rename(columns={"formal": "formal_query", "informal": "generated_informal_query"})


def load_miracl_pairs(data_dir: str = DATA_DIR) -> List[dict]:
    with open(os.path.join(data_dir, "miracl_pairs.json")) as f:
        return json.load(f)


def load_qrels(data_dir: str = DATA_DIR) -> Dict[str, set]:
    """{query_id: {relevant doc_id, ...}} (binary relevance)."""
    with open(os.path.join(data_dir, "qrels.json")) as f:
        rows = json.load(f)
    qrels = defaultdict(set)
    for r in rows:
        qrels[r["query_id"]].add(r["relevant_doc"])
    return dict(qrels)


def load_corpus(data_dir: str = DATA_DIR):
    """Returns (doc_ids, passages) as parallel lists."""
    with open(os.path.join(data_dir, "corpus.json")) as f:
        rows = json.load(f)
    return [r["doc_id"] for r in rows], [r["passage"] for r in rows]


# --------------------------------------------------------------------------- #
# Joins and splits
# --------------------------------------------------------------------------- #
def build_triples(data_dir: str = DATA_DIR, hard_negs: Dict[str, List[str]] = None) -> List[dict]:
    """Join train_pairs.csv with miracl_pairs.json on query_id.

    One record per (query, positive doc):
    (informal_query, formal_query, positive_doc_id, positive_doc_text, negative_doc_ids)
    miracl_pairs.json carries no negatives, so these come from mined hard negatives.
    """
    tp = load_train_pairs(data_dir).set_index("query_id")
    triples = []
    for p in load_miracl_pairs(data_dir):
        qid = p["query_id"]
        if qid not in tp.index:
            continue
        triples.append({
            "query_id": qid,
            "informal_query": tp.at[qid, "generated_informal_query"],
            "formal_query": tp.at[qid, "formal_query"],
            "positive_doc_id": p["positive_doc_id"],
            "positive_doc_text": p["positive_text"],
            "negative_doc_ids": (hard_negs or {}).get(qid, []),
        })
    return triples


def make_splits(query_ids: List[str], dev_frac: float = 0.10, test_frac: float = 0.15, seed: int = 42) -> Dict[str, List[str]]:
    """Disjoint query-level split, so no test query (formal or informal) is seen in training."""
    qids = sorted(set(query_ids), key=int)
    random.Random(seed).shuffle(qids)
    n_test, n_dev = round(len(qids) * test_frac), round(len(qids) * dev_frac)
    return {
        "test": sorted(qids[:n_test], key=int),
        "dev": sorted(qids[n_test : n_test + n_dev], key=int),
        "train": sorted(qids[n_test + n_dev :], key=int),
    }


# --------------------------------------------------------------------------- #
# Cached embeddings
# --------------------------------------------------------------------------- #
class EmbeddingStore:
    """All frozen BGE-M3 embeddings (produced by prepare.py) as tensors on `device`."""

    def __init__(self, cache_dir: str = CACHE_DIR, device=None, load_corpus: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with open(os.path.join(cache_dir, "query_ids.json")) as f:
            self.query_ids = json.load(f)
        self.qid2row = {q: i for i, q in enumerate(self.query_ids)}
        self.formal = self._load(os.path.join(cache_dir, "formal_emb.npy"))
        self.informal = self._load(os.path.join(cache_dir, "informal_emb.npy"))
        with open(os.path.join(cache_dir, "corpus_ids.json")) as f:
            self.doc_ids = json.load(f)
        self.doc2row = {d: i for i, d in enumerate(self.doc_ids)}
        self.corpus = self._load(os.path.join(cache_dir, "corpus_emb.npy")) if load_corpus else None

    def _load(self, path: str) -> torch.Tensor:
        return torch.from_numpy(np.load(path, mmap_mode="r")[:].astype(np.float16)).to(self.device)


class TrainQueryDataset(Dataset):
    """One item per training query; a positive and a hard negative are re-sampled every epoch."""

    def __init__(self, triples: List[dict], train_qids: List[str], store: EmbeddingStore,
                 hn_range=(10, 100), seed: int = 42):
        train = set(train_qids)
        self.store = store
        self.positives = defaultdict(list)  # qid -> [corpus row]
        self.negatives = {}
        for t in triples:
            if t["query_id"] in train and t["positive_doc_id"] in store.doc2row:
                self.positives[t["query_id"]].append(store.doc2row[t["positive_doc_id"]])
                self.negatives[t["query_id"]] = [store.doc2row[d] for d in t["negative_doc_ids"][hn_range[0]:hn_range[1]]
                                                 if d in store.doc2row]
        self.qids = sorted(self.positives, key=int)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, i):
        qid = self.qids[i]
        negs = self.negatives[qid]
        return {
            "q_row": self.store.qid2row[qid],
            "pos_row": self.rng.choice(self.positives[qid]),
            "neg_row": self.rng.choice(negs) if negs else self.rng.randrange(len(self.store.doc_ids)),
            "pos_set": self.positives[qid],
        }

    def collate(self, items: List[dict]) -> Dict[str, torch.Tensor]:
        s = self.store
        q = torch.tensor([it["q_row"] for it in items], device=s.device)
        cand = torch.tensor([it["pos_row"] for it in items] + [it["neg_row"] for it in items], device=s.device)
        # false_neg[i, j] = candidate j is a labelled positive of query i (but not its own target).
        false_neg = torch.zeros(len(items), len(cand), dtype=torch.bool)
        cand_list = cand.tolist()
        for i, it in enumerate(items):
            pos = set(it["pos_set"])
            for j, c in enumerate(cand_list):
                if j != i and c in pos:
                    false_neg[i, j] = True
        return {
            "z_inf": s.informal[q].float(),
            "z_form": s.formal[q].float(),
            "z_cand": s.corpus[cand].float(),  # [positives (B) ; hard negatives (B)]
            "false_neg": false_neg.to(s.device),
        }

    def loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=self.collate)
