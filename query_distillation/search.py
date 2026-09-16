"""Search with the Query distillation adapter and score each query with nDCG@10, MRR@10 and Recall@100.

Interactive (BGE-M3, adapter and corpus load once, ~40s):
  python3 search.py
    query> probolinggo makanan khas apa sih   dataset queries (informal or formal text) are recognised and scored
    query> :id 4502                           run dataset query 4502 (its informal version)
    query> :random                            random test query (:random dev / :random train)
    query> gunung paling tinggi apaan ya      your own query: results only...
    query> :rel 20334#0 20334#4               ...then mark the relevant passages to score it
    query> :k 10 | :full | :help | :quit

One-shot:
  python3 search.py --qid 4502
  python3 search.py --query "presiden pertama indonesia siapa sih" --relevant 40928#2 2834#0 --json

Metrics use the top-100 ranking and only relevant passages that exist in the corpus (same as evaluate.py).
Queries from the train split were seen during training, so their scores are optimistic.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
from typing import Dict, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset import EmbeddingStore, load_corpus, load_qrels, load_train_pairs  # noqa: E402
from eval import METRICS, dense_search, per_query_metrics  # noqa: E402
from evaluate import adapt, load_adapter  # noqa: E402
from model import BGEM3Encoder  # noqa: E402

DEPTH = 100
HELP = """commands:
  <text>              search; dataset queries (informal or formal text) are recognised and scored automatically
  :id <query_id>      run a dataset query's informal version, e.g. :id 4502
  :random [split]     random dataset query from test (default), dev or train
  :rel <doc_id> ...   mark relevant passages for the last query and score it
  :k <N>              passages to show (1-100)
  :full               toggle full passage text
  :help | :quit"""


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


class DistillationRetriever:
    def __init__(self, ckpt: str, cache_dir: str, data_dir: str, model_name: str = "BAAI/bge-m3"):
        self.store = EmbeddingStore(cache_dir)
        self.device = self.store.device
        self.encoder = BGEM3Encoder(model_name, device=str(self.device))
        self.adapter = load_adapter(ckpt, self.device)
        doc_ids, self.passages = load_corpus(data_dir)
        assert doc_ids == self.store.doc_ids, "corpus.json order differs from the cached corpus embeddings"
        self.in_corpus = set(self.store.doc2row)
        self.qrels = load_qrels(data_dir)
        self.pairs = load_train_pairs(data_dir).set_index("query_id")
        with open(os.path.join(cache_dir, "splits.json")) as f:
            self.split_of = {q: name for name, qids in json.load(f).items() for q in qids}
        self.lookup: Dict[str, List[str]] = {}
        for qid, row in self.pairs.iterrows():
            for col in ("generated_informal_query", "formal_query"):
                self.lookup.setdefault(normalise(row[col]), []).append(qid)

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        return torch.from_numpy(self.encoder.encode([text], max_length=64, progress=False)).to(self.device).float()

    def search(self, z: torch.Tensor):
        scores, idx = dense_search(z, self.store.corpus, DEPTH)
        return scores[0].tolist(), idx[0].tolist()

    def relevant_for(self, qid: str) -> set:
        return self.qrels.get(qid, set()) & self.in_corpus

    def score(self, rows: List[int], relevant: set) -> Dict[str, Optional[float]]:
        ranked = [self.store.doc_ids[j] for j in rows]
        m = {k: float(v[0]) for k, v in per_query_metrics({"q": ranked}, {"q": relevant}).items()}
        m["first_relevant_rank"] = next((r + 1 for r, d in enumerate(ranked) if d in relevant), None)
        return m


class Session:
    def __init__(self, retriever: DistillationRetriever, k: int, full: bool):
        self.r, self.k, self.full = retriever, k, full
        self.last = None

    def run(self, text: str, qid: Optional[str] = None, relevant: Optional[set] = None, show: bool = True) -> dict:
        r = self.r
        matches = [qid] if qid else r.lookup.get(normalise(text), [])
        qid = matches[0] if matches else None
        z_base = r.encode(text)
        base_scores, base_rows = r.search(z_base)
        dist_scores, dist_rows = r.search(adapt(r.adapter, z_base))
        result = {"query": text, "query_id": qid, "split": r.split_of.get(qid) if qid else None,
                  "base": {"scores": base_scores, "rows": base_rows}, "distill": {"scores": dist_scores, "rows": dist_rows}}
        if qid:
            result["formal_query"] = r.pairs.at[qid, "formal_query"]
            result["informal_query"] = r.pairs.at[qid, "generated_informal_query"]
            result["other_matching_ids"] = matches[1:]
            if relevant is None:
                relevant = r.relevant_for(qid)
                if normalise(text) != normalise(result["formal_query"]):
                    _, formal_rows = r.search(r.encode(result["formal_query"]))
                    result["formal"] = {"rows": formal_rows}
        self.last = result
        self.rescore(relevant or set(), show)
        return result

    def rescore(self, relevant: set, show: bool = True):
        res, r = self.last, self.r
        res["relevant"] = sorted(relevant)
        res["metrics"] = {}
        if relevant:
            res["metrics"]["base"] = r.score(res["base"]["rows"], relevant)
            res["metrics"]["distill"] = r.score(res["distill"]["rows"], relevant)
            if "formal" in res:
                res["metrics"]["formal"] = r.score(res["formal"]["rows"], relevant)
        if show:
            self.print_result()

    def print_result(self):
        res, r = self.last, self.r
        print(f"\n{'=' * 100}\nquery: {res['query']}")
        if res["query_id"]:
            seen = "  ⚠ seen during training, scores are optimistic" if res["split"] == "train" else ""
            print(f"dataset query {res['query_id']} [{res['split']} split]{seen}")
            print(f"  informal: {res['informal_query']}\n  formal  : {res['formal_query']}")
            if res["other_matching_ids"]:
                print(f"  (the same text also appears as query {', '.join(res['other_matching_ids'])})")
        relevant = set(res["relevant"])
        if res["metrics"]:
            print(f"\n  {'metrics (' + str(len(relevant)) + ' relevant passage(s))':<38} {'nDCG@10':>8} {'MRR@10':>8} {'R@100':>8} {'1st relevant':>13}")
            labels = {"base": "base BGE-M3", "distill": "Query distillation", "formal": "base BGE-M3 on formal version"}
            for key, m in res["metrics"].items():
                rank = m["first_relevant_rank"] or f">{DEPTH}"
                print(f"    {labels[key]:<36} {m['nDCG@10']:>8.4f} {m['MRR@10']:>8.4f} {m['Recall@100']:>8.4f} {rank:>13}")
        elif res["query_id"]:
            print("\n  no relevant passage for this query exists in the corpus, so it cannot be scored")
        else:
            print("\n  no relevance labels for this text: mark relevant passages with  :rel <doc_id> ...  to score it")

        base_rank = {j: i + 1 for i, j in enumerate(res["base"]["rows"])}
        top = res["distill"]["rows"][: self.k]
        print(f"\n  top-{self.k} Query distillation   (✓ relevant, base #N = rank under base BGE-M3)")
        for i, j in enumerate(top):
            doc = r.store.doc_ids[j]
            mark = "✓" if doc in relevant else ("·" if relevant else " ")
            moved = f"base #{base_rank[j]}" if j in base_rank else f"base >{DEPTH}"
            print(f"\n  {i + 1:>2}. {mark} {res['distill']['scores'][i]:.4f}  [{doc}]  {moved}")
            text = " ".join(r.passages[j].split())
            if not self.full:
                text = textwrap.shorten(text, width=300, placeholder=" …")
            print(textwrap.fill(text, width=96, initial_indent="      ", subsequent_indent="      "))
        dropped = [j for j in res["base"]["rows"][: self.k] if j not in set(top)]
        for j in dropped:
            doc = r.store.doc_ids[j]
            print(f"\n  dropped from base top-{self.k}: base #{base_rank[j]} {'✓' if doc in relevant else ' '} [{doc}] "
                  f"{textwrap.shorten(' '.join(r.passages[j].split()), width=70, placeholder=' …')}")

    def command(self, line: str) -> bool:
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        r = self.r
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd == "id":
            if arg in r.pairs.index:
                self.run(r.pairs.at[arg, "generated_informal_query"], qid=arg)
            else:
                print(f"unknown query_id {arg!r}")
        elif cmd == "random":
            split = arg or "test"
            qids = [q for q, s in r.split_of.items() if s == split]
            if qids:
                q = random.choice(qids)
                self.run(r.pairs.at[q, "generated_informal_query"], qid=q)
            else:
                print("split must be train, dev or test")
        elif cmd == "rel":
            if self.last is None:
                print("search something first")
            else:
                docs = set(arg.replace(",", " ").split())
                unknown = sorted(docs - r.in_corpus)
                if unknown:
                    print(f"ignored (not in corpus): {' '.join(unknown)}")
                self.last.pop("formal", None)
                self.rescore(docs & r.in_corpus)
        elif cmd == "k" and arg.isdigit() and 0 < int(arg) <= DEPTH:
            self.k = int(arg)
            print(f"k = {self.k}")
        elif cmd == "full":
            self.full = not self.full
            print(f"full passage text = {self.full}")
        else:
            print(HELP)
        return True


def to_json(res: dict, r: DistillationRetriever, k: int) -> dict:
    ids = r.store.doc_ids
    out = {key: res.get(key) for key in ("query", "query_id", "split", "formal_query", "relevant", "metrics")}
    for name in ("distill", "base"):
        out[f"top_{name}"] = [{"rank": i + 1, "score": round(res[name]["scores"][i], 4), "doc_id": ids[j],
                               "relevant": ids[j] in set(res["relevant"]), "passage": r.passages[j]}
                              for i, j in enumerate(res[name]["rows"][:k])]
    return out


def main():
    ap = argparse.ArgumentParser(description="Query distillation search with per-query nDCG@10 / MRR@10 / Recall@100")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "score", "best"))
    ap.add_argument("--query", action="append", help="Query text (repeatable); runs once and exits")
    ap.add_argument("--qid", action="append", help="Dataset query_id (repeatable); runs its informal version")
    ap.add_argument("--relevant", nargs="*", help="Relevant doc_ids for a --query that is not in the dataset")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--json", action="store_true", help="One-shot mode: print results as JSON")
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()

    print(f"Loading BGE-M3, adapter ({args.ckpt}) and corpus (~40s)...", file=sys.stderr)
    session = Session(DistillationRetriever(args.ckpt, args.cache_dir, args.data_dir), args.k, args.full)

    if args.query or args.qid:
        relevant = set(args.relevant) & session.r.in_corpus if args.relevant else None
        results = [session.run(session.r.pairs.at[q, "generated_informal_query"], qid=q, show=not args.json)
                   for q in args.qid or []]
        results += [session.run(q, relevant=relevant, show=not args.json) for q in args.query or []]
        if args.json:
            json.dump([to_json(res, session.r, args.k) for res in results], sys.stdout, indent=2, ensure_ascii=False)
            print()
        return

    print(f"\nReady.\n{HELP}")
    while True:
        try:
            line = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith(":"):
            if not session.command(line):
                break
        else:
            session.run(line)


if __name__ == "__main__":
    main()
