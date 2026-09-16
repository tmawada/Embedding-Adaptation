"""Type a query, get the retrieved passages (BGE-M3 + Query-DANN adapter over the 500k MIRACL-id corpus).

Interactive (model and corpus load once, then every query is instant):
  python3 search.py
    query> presiden pertama indonesia siapa sih
    query> :k 10            show 10 passages
    query> :mode both       adapted | base | both (adapted list annotated with base-model rank)
    query> :full            toggle full passage text
    query> :quit

One-shot:
  python3 search.py --query "gunung paling tinggi sedunia apaan ya" --k 5
  python3 search.py --query "..." --query "..." --json > results.json
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import textwrap
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from inference import QueryDANNRetriever

MODES = ("adapted", "base", "both")
HELP = "commands: :k N | :mode adapted|base|both | :full | :help | :quit"


def retrieve(retriever: QueryDANNRetriever, queries: List[str], k: int) -> List[Dict[str, list]]:
    """Top-k passages for each query, with and without the adapter."""
    z_base = retriever.encode(queries)
    z_adapt = retriever.head.adapt(z_base)
    results = [{} for _ in queries]
    for name, z in (("adapted", z_adapt), ("base", z_base)):
        scores, rows = retriever.search(z, k)
        for i in range(len(queries)):
            results[i][name] = [{"rank": r + 1, "score": round(scores[i][r], 4),
                                 "doc_id": retriever.store.doc_ids[rows[i][r]],
                                 "passage": retriever.passages[rows[i][r]]} for r in range(k)]
    return results


def print_results(query: str, result: Dict[str, list], mode: str, full: bool):
    print(f"\n{'=' * 100}\nquery: {query}")
    shown = "adapted" if mode == "both" else mode
    base_rank = {d["doc_id"]: d["rank"] for d in result["base"]}
    label = {"adapted": "Query-DANN (adapted)", "base": "base BGE-M3"}[shown]
    print(f"{label} — top {len(result[shown])}" + ("   [base #N = rank under base BGE-M3]" if mode == "both" else ""))
    for d in result[shown]:
        note = ""
        if mode == "both":
            note = f"  [base #{base_rank[d['doc_id']]}]" if d["doc_id"] in base_rank else "  [new: not in base top-k]"
        print(f"\n  {d['rank']:>2}. score {d['score']:.4f}  doc {d['doc_id']}{note}")
        text = " ".join(d["passage"].split())
        if not full:
            text = textwrap.shorten(text, width=320, placeholder=" …")
        print(textwrap.fill(text, width=96, initial_indent="      ", subsequent_indent="      "))
    if mode == "both":
        dropped = [d for d in result["base"] if d["doc_id"] not in {a["doc_id"] for a in result["adapted"]}]
        for d in dropped:
            print(f"\n  dropped by adapter: base #{d['rank']} doc {d['doc_id']} — "
                  f"{textwrap.shorten(' '.join(d['passage'].split()), width=80, placeholder=' …')}")


def interactive(retriever: QueryDANNRetriever, k: int, mode: str, full: bool):
    print(f"\nReady. Type a query and press Enter. {HELP}")
    while True:
        try:
            line = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith(":"):
            cmd, _, arg = line[1:].partition(" ")
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "k" and arg.strip().isdigit() and 0 < int(arg) <= 100:
                k = int(arg)
                print(f"k = {k}")
            elif cmd == "mode" and arg.strip() in MODES:
                mode = arg.strip()
                print(f"mode = {mode}")
            elif cmd == "full":
                full = not full
                print(f"full passage text = {full}")
            else:
                print(HELP + "   (k must be 1..100)")
            continue
        print_results(line, retrieve(retriever, [line], k)[0], mode, full)


def main():
    ap = argparse.ArgumentParser(description="Retrieve passages for a query with BGE-M3 + Query-DANN")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "v2_dann_g1.0", "best"))
    ap.add_argument("--query", action="append", help="Run these queries and exit (repeatable)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", default="adapted", choices=MODES)
    ap.add_argument("--full", action="store_true", help="Print full passage text instead of a 320-char preview")
    ap.add_argument("--json", action="store_true", help="With --query: print results as JSON")
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()

    print("Loading BGE-M3, adapter and corpus (~40s)...", file=sys.stderr)
    retriever = QueryDANNRetriever(args.ckpt, args.cache_dir, args.data_dir)

    if not args.query:
        interactive(retriever, args.k, args.mode, args.full)
        return
    results = retrieve(retriever, args.query, args.k)
    if args.json:
        json.dump([{"query": q, **r} for q, r in zip(args.query, results)], sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        for q, r in zip(args.query, results):
            print_results(q, r, args.mode, args.full)


if __name__ == "__main__":
    main()
