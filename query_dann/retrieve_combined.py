"""Retrieve passages combining LLM normalization and DANN V2 (linear adapter).

Pipeline:
  1. Take an informal Indonesian query
  2. Normalize it to formal via Gemini LLM
  3. Encode with frozen BGE-M3
  4. Apply DANN V2 adapter to informal embedding
  5. Search corpus with: base formal, base informal, DANN adapted informal, LLM normalized formal

Usage:
  python3 query_dann/retrieve_combined.py --query "presiden pertama indonesia siapa sih"
  python3 query_dann/retrieve_combined.py --query "presiden pertama indonesia siapa sih" --k 10
  python3 query_dann/retrieve_combined.py --query "presiden pertama indonesia siapa sih" --json
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import textwrap

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset import CACHE_DIR, DATA_DIR, EmbeddingStore, load_corpus
from eval import dense_search, load_head
from model import BGEM3Encoder

LLM_CACHE_DIR = os.path.join(ROOT, "llm_rewrite", "cache")

SEARCH_K = 100


class CombinedRetriever:
    """BGE-M3 + DANN V2 adapter + LLM normalization."""

    def __init__(self, ckpt: str, cache_dir: str = CACHE_DIR, data_dir: str = DATA_DIR,
                 model_name: str = "BAAI/bge-m3"):
        self.store = EmbeddingStore(cache_dir)
        self.device = self.store.device
        self.encoder = BGEM3Encoder(model_name, device=str(self.device))
        self.head = load_head(ckpt, self.device)
        doc_ids, self.passages = load_corpus(data_dir)
        assert doc_ids == self.store.doc_ids, "corpus.json order differs from cached corpus embeddings"

    @torch.no_grad()
    def encode(self, texts):
        z = torch.from_numpy(self.encoder.encode(texts, max_length=64, progress=False)).to(self.device).float()
        return z

    def search(self, z: torch.Tensor, k: int = SEARCH_K):
        scores, idx = dense_search(z, self.store.corpus, k)
        return scores.tolist(), idx.tolist()


def load_cached_rewrites() -> dict:
    """Load cached LLM rewrites from llm_rewrite/cache/*.jsonl."""
    cache = {}
    for fname in os.listdir(LLM_CACHE_DIR):
        if fname.endswith(".jsonl") and "formal" not in fname:
            with open(os.path.join(LLM_CACHE_DIR, fname)) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        cache[rec["informal"].strip().lower()] = rec["normalized"]
    return cache


def normalize_query(query: str, cached: dict = None, data_dir: str = DATA_DIR,
                    cache_dir: str = CACHE_DIR) -> tuple:
    """Normalize informal query to formal. Tries cache first, then Gemini API.

    Live calls mirror llm_rewrite/normalize.py: 3 few-shot examples from the
    train split, no thinking (so max_output_tokens is fully usable for output).
    """
    if cached and query.strip().lower() in cached:
        return cached[query.strip().lower()], 0.0
    try:
        import random
        from dataset import load_train_pairs
        from llm_rewrite.normalize import load_api_key, call_gemini, RateLimiter, build_prompt
        api_key = load_api_key()
        tp = load_train_pairs(data_dir).set_index("query_id")
        with open(os.path.join(cache_dir, "splits.json")) as f:
            train_qids = json.load(f)["train"]
        shots = [(tp.at[q, "generated_informal_query"], tp.at[q, "formal_query"])
                 for q in random.Random(0).sample(train_qids, 3)]
        prompt = build_prompt(query, shots)
        limiter = RateLimiter(rpm=15)
        text, latency, _ = call_gemini(api_key, "gemini-2.5-flash", prompt, limiter,
                                       max_output_tokens=128, no_thinking=True)
        return text.splitlines()[0].strip(), latency
    except SystemExit:
        raise SystemExit(
            "No Gemini API key found and query not in cache. "
            "Set GEMINI_API_KEY or use a query from the test set."
        )


def print_topk(title: str, scores, rows, retriever: CombinedRetriever, k: int):
    print(f"\n  {title}")
    for r in range(min(k, len(rows))):
        doc_id = retriever.store.doc_ids[rows[r]]
        print(f"   {r + 1:2d}. {scores[r]:.3f} [{doc_id:>11}] {textwrap.shorten(' '.join(retriever.passages[rows[r]].split()), width=80, placeholder=' ...')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Informal query to retrieve")
    ap.add_argument("--k", type=int, default=10, help="Top-k results to show")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "v2_linear_g1.0", "best"))
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    args = ap.parse_args()

    print("Loading BGE-M3, DANN V2 linear adapter, and corpus...", file=sys.stderr)
    retriever = CombinedRetriever(args.ckpt, args.cache_dir, args.data_dir)

    # Step 1: LLM normalization (cache first, then Gemini API)
    print(f"\n[1/3] Normalizing query via LLM...", file=sys.stderr)
    cached = load_cached_rewrites()
    formal_query, latency = normalize_query(args.query, cached)
    print(f"  informal : {args.query}", file=sys.stderr)
    print(f"  formal   : {formal_query}  ({latency:.2f}s)", file=sys.stderr)

    # Step 2: Encode all query variants
    print(f"\n[2/3] Encoding with BGE-M3...", file=sys.stderr)
    queries = {
        "informal": args.query,
        "formal": formal_query,
    }
    z_informal = retriever.encode([args.query])
    z_formal = retriever.encode([formal_query])

    # Step 3: DANN adaptation
    print(f"\n[3/3] Applying DANN V2 linear adapter...", file=sys.stderr)
    z_adapted = retriever.head.adapt(z_informal)

    # Step 4: Search
    print(f"\nSearching corpus ({len(retriever.store.doc_ids):,} passages)...", file=sys.stderr)
    (s_base_inf, r_base_inf) = retriever.search(z_informal, args.k)
    (s_adapted, r_adapted) = retriever.search(z_adapted, args.k)
    (s_base_form, r_base_form) = retriever.search(z_formal, args.k)

    # Cosine similarities
    cos_base_adapted = float((z_informal[0].detach() * z_adapted[0].detach()).sum())
    cos_inf_formal = float((z_informal[0].detach() * z_formal[0].detach()).sum())
    cos_adapted_formal = float((z_adapted[0].detach() * z_formal[0].detach()).sum())

    # Output
    if args.json:
        results = {
            "query_informal": args.query,
            "query_formal": formal_query,
            "llm_latency_s": round(latency, 3),
            "cos_similarities": {
                "informal_vs_formal": round(cos_inf_formal, 4),
                "adapted_vs_formal": round(cos_adapted_formal, 4),
                "base_vs_adapted": round(cos_base_adapted, 4),
            },
            "top_k": {
                "base_informal": [{"rank": r + 1, "score": round(s_base_inf[0][r], 4),
                                   "doc_id": retriever.store.doc_ids[r_base_inf[0][r]],
                                   "passage": retriever.passages[r_base_inf[0][r]]} for r in range(args.k)],
                "dann_adapted": [{"rank": r + 1, "score": round(s_adapted[0][r], 4),
                                  "doc_id": retriever.store.doc_ids[r_adapted[0][r]],
                                  "passage": retriever.passages[r_adapted[0][r]]} for r in range(args.k)],
                "llm_formal": [{"rank": r + 1, "score": round(s_base_form[0][r], 4),
                                "doc_id": retriever.store.doc_ids[r_base_form[0][r]],
                                "passage": retriever.passages[r_base_form[0][r]]} for r in range(args.k)],
            }
        }
        json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        print(f"\n{'=' * 100}")
        print(f"Query (informal): {args.query}")
        print(f"Query (formal)  : {formal_query}")
        print(f"LLM latency: {latency:.2f}s")
        print(f"\nCosine similarities:")
        print(f"  informal vs formal      : {cos_inf_formal:.4f}")
        print(f"  adapted vs formal       : {cos_adapted_formal:.4f}  (adapter moves informal toward formal)")
        print(f"  base informal vs adapted: {cos_base_adapted:.4f}")

        print_topk(f"Top-{args.k} base BGE-M3 (informal):", s_base_inf[0], r_base_inf[0], retriever, args.k)
        print_topk(f"Top-{args.k} DANN V2 linear (adapted):", s_adapted[0], r_adapted[0], retriever, args.k)
        print_topk(f"Top-{args.k} base BGE-M3 (LLM formal):", s_base_form[0], r_base_form[0], retriever, args.k)

        # Overlap analysis
        base_set = set(retriever.store.doc_ids[r_base_inf[0][r]] for r in range(args.k))
        adapted_set = set(retriever.store.doc_ids[r_adapted[0][r]] for r in range(args.k))
        formal_set = set(retriever.store.doc_ids[r_base_form[0][r]] for r in range(args.k))

        print(f"\nOverlap analysis (top-{args.k}):")
        print(f"  base informal ∩ adapted    : {len(base_set & adapted_set)}/{args.k}")
        print(f"  base informal ∩ LLM formal : {len(base_set & formal_set)}/{args.k}")
        print(f"  adapted ∩ LLM formal       : {len(adapted_set & formal_set)}/{args.k}")


if __name__ == "__main__":
    main()
