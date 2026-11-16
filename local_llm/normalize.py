"""Normalize informal Indonesian queries using local LLM (Ollama).

Reuses the exact same prompt template as the Gemini baseline for fair comparison.
Results are cached as JSONL (one record per query, resumable) with per-call latency.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # bge_dann root
sys.path.insert(0, ROOT)

from preprocessing.dataset import load_train_pairs  # noqa: E402
from local_llm.client import OllamaClient, GenerationResult  # noqa: E402


# Exact same instruction as Gemini baseline (methods/llm_rewrite/normalize.py)
PROMPT_VERSION = "id-normalise-v1"

INSTRUCTION = """Tugas: ubah pertanyaan informal (bahasa gaul/sehari-hari) menjadi pertanyaan formal bahasa Indonesia baku untuk sistem pencarian dokumen.

Aturan:
- JANGAN menjawab pertanyaannya.
- Jangan menambah atau menghilangkan informasi.
- Pertahankan nama entitas, angka, dan ejaan nama asing apa adanya.
- Keluarkan HANYA pertanyaan formalnya, satu baris, tanpa penjelasan."""


@dataclass
class NormalizationConfig:
    """Configuration for normalization."""
    model_name: str
    endpoint: str
    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int
    repeat_penalty: float
    seed: int
    few_shot_examples: int
    batch_size: int
    max_concurrent: int
    prompt_version: str


def load_config(config_path: str) -> NormalizationConfig:
    """Load normalization configuration from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_cfg = config.get("model", {})
    norm_cfg = config.get("normalization", {})
    
    return NormalizationConfig(
        model_name=model_cfg.get("name", "qwen2.5:3b-instruct-q4_K_M"),
        endpoint=model_cfg.get("endpoint", "http://localhost:11434"),
        temperature=model_cfg.get("temperature", 0.0),
        top_p=model_cfg.get("top_p", 1.0),
        top_k=model_cfg.get("top_k", 1),
        max_output_tokens=model_cfg.get("max_output_tokens", 128),
        repeat_penalty=model_cfg.get("repeat_penalty", 1.0),
        seed=model_cfg.get("seed", 42),
        few_shot_examples=norm_cfg.get("few_shot_examples", 3),
        batch_size=norm_cfg.get("batch_size", 1),
        max_concurrent=norm_cfg.get("max_concurrent", 1),
        prompt_version=norm_cfg.get("prompt_version", PROMPT_VERSION),
    )


def build_prompt(informal: str, shots: List[tuple]) -> str:
    """Build few-shot prompt for a single query (matches Gemini format exactly)."""
    lines = [INSTRUCTION, ""]
    for inf, form in shots:
        lines += [f"Informal: {inf}", f"Formal: {form}", ""]
    lines += [f"Informal: {informal}", "Formal:"]
    return "\n".join(lines)


def build_batch_prompt(informals: List[str], shots: List[tuple]) -> str:
    """Build few-shot prompt for batch of queries."""
    lines = [INSTRUCTION, ""]
    for inf, form in shots:
        lines += [f"Informal: {inf}", f"Formal: {form}", ""]
    
    numbered = "\n".join(f"{i + 1}. {inf}" for i, inf in enumerate(informals))
    lines += [
        f"\nUbah setiap pertanyaan berikut. Keluarkan daftar bernomor dengan jumlah baris "
        f"yang sama ({len(informals)} baris), format `nomor. pertanyaan formal`, tanpa teks lain.\n\n"
        f"{numbered}"
    ]
    return "\n".join(lines)


def parse_batch_response(text: str, expected_count: int) -> List[str]:
    """Parse numbered response from batch generation."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    parsed = []
    for l in lines:
        head, _, rest = l.partition(".")
        if head.strip().isdigit() and rest.strip():
            parsed.append(rest.strip())
    
    if len(parsed) != expected_count:
        # Fallback: try to extract each line as a separate query
        if len(lines) == expected_count:
            return [l.split(".", 1)[1].strip() if "." in l else l for l in lines]
        raise RuntimeError(f"Expected {expected_count} lines, parsed {len(parsed)}: {parsed}")
    
    return parsed


class LocalLLMNormalizer:
    """Normalize informal queries using local LLM via Ollama."""

    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.client = OllamaClient(
            endpoint=config.endpoint,
            model=config.model_name,
        )
        self.options = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "num_predict": config.max_output_tokens,
            "repeat_penalty": config.repeat_penalty,
            "seed": config.seed,
        }

    def normalize_single(self, informal: str, shots: List[tuple]) -> GenerationResult:
        """Normalize a single informal query."""
        prompt = build_prompt(informal, shots)
        return self.client.generate(prompt, options=self.options)

    def normalize_batch(self, informals: List[str], shots: List[tuple]) -> List[GenerationResult]:
        """Normalize a batch of informal queries."""
        if len(informals) == 1:
            result = self.normalize_single(informals[0], shots)
            return [result]
        
        prompt = build_batch_prompt(informals, shots)
        result = self.client.generate(prompt, options=self.options)
        
        try:
            parsed = parse_batch_response(result.text, len(informals))
        except RuntimeError as e:
            # Fallback to individual requests
            print(f"  Batch parsing failed ({e}); falling back to per-query", flush=True)
            return [self.normalize_single(inf, shots) for inf in informals]
        
        # Distribute latency across queries
        per_query_latency = result.latency_s / len(informals)
        per_query_tokens = result.total_tokens // len(informals) if result.total_tokens else 0
        
        return [
            GenerationResult(
                text=norm,
                latency_s=per_query_latency,
                prompt_tokens=result.prompt_tokens // len(informals) if result.prompt_tokens else 0,
                completion_tokens=per_query_tokens,
                total_tokens=per_query_tokens,
                model=result.model,
                done=result.done,
            )
            for norm in parsed
        ]

    def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        return self.client.health_check()


def main():
    ap = argparse.ArgumentParser(description="Normalize informal Indonesian queries using local LLM (Ollama)")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--queries", default="informal", choices=["informal", "formal", "informal_varied"],
                    help="Which query text to normalize")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "qwen2.5_3b.yaml"),
                    help="Path to YAML configuration")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "methods", "cache"), help="For splits.json")
    ap.add_argument("--out", default=None, help="Output JSONL (default: local_llm/cache/<model>_<split>.jsonl)")
    ap.add_argument("--shots", type=int, default=None, help="Few-shot examples (overrides config)")
    ap.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    ap.add_argument("--concurrency", type=int, default=None, help="Max concurrent requests (overrides config)")
    ap.add_argument("--limit", type=int, default=None, help="Only first N queries (smoke test)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for few-shot selection")
    ap.add_argument("--list_models", action="store_true", help="List available models and exit")
    args = ap.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.shots is not None:
        config.few_shot_examples = args.shots
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.concurrency is not None:
        config.max_concurrent = args.concurrency

    # Initialize normalizer
    normalizer = LocalLLMNormalizer(config)
    
    if args.list_models:
        models = normalizer.client.list_models()
        print(f"{'model':<42} {'size (GB)':>10}")
        for m in models:
            print(f"{m.name:<42} {m.size / 1e9:>10.2f}")
        return

    # Health check
    if not normalizer.health_check():
        raise SystemExit(f"Cannot connect to Ollama at {config.endpoint}. Is Ollama running?")

    # Load data
    tp = load_train_pairs(args.data_dir).set_index("query_id")
    column = {"informal": "generated_informal_query", "formal": "formal_query",
              "informal_varied": "informal_varied"}[args.queries]
    
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        splits = json.load(f)
    qids = splits[args.split][: args.limit]
    
    # Select few-shot examples from TRAIN split only
    train_qids = splits["train"]
    random.seed(args.seed)
    shot_qids = random.sample(train_qids, min(config.few_shot_examples, len(train_qids)))
    shots = [(tp.at[q, "generated_informal_query"], tp.at[q, "formal_query"]) for q in shot_qids]

    # Output path
    suffix = {"informal": "", "formal": "_formal", "informal_varied": "_varied"}[args.queries]
    model_safe = config.model_name.replace(":", "_").replace("/", "_")
    out_path = args.out or os.path.join(HERE, "cache", f"{model_safe}_{args.split}{suffix}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load existing cache
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(l)["query_id"]: 1 for l in f if l.strip()}
    todo = [q for q in qids if q not in done]
    
    print(f"[{config.model_name}] split={args.split} queries={len(qids)} cached={len(done)} to do={len(todo)} "
          f"batch_size={config.batch_size} concurrency={config.max_concurrent} shots={config.few_shot_examples} "
          f"-> {os.path.relpath(out_path, ROOT)}")
    if not todo:
        return

    write_lock = threading.Lock()
    out_file = open(out_path, "a")
    stats = {"ok": 0, "failed": 0}
    stop = threading.Event()

    def record(qid: str, informal_text: str, normalized: str, result: GenerationResult, batch_size: int):
        with write_lock:
            out_file.write(json.dumps({
                "query_id": qid,
                "queries": args.queries,
                "informal": informal_text,
                "normalized": normalized,
                "latency_s": round(result.latency_s, 4),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "model": result.model,
                "prompt_version": config.prompt_version,
                "shots": config.few_shot_examples,
                "batch_size": batch_size,
                "temperature": config.temperature,
            }, ensure_ascii=False) + "\n")
            out_file.flush()
            stats["ok"] += 1
            n = stats["ok"] + stats["failed"]
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} done (failed {stats['failed']})", flush=True)

    def process_chunk(chunk: List[str]):
        if stop.is_set():
            return
        try:
            informal_texts = [tp.at[q, column] for q in chunk]
            results = normalizer.normalize_batch(informal_texts, shots)
            for qid, inf_text, res in zip(chunk, informal_texts, results):
                # Take first line only
                norm_clean = res.text.splitlines()[0].strip()
                record(qid, inf_text, norm_clean, res, len(chunk))
        except Exception as e:
            if not stop.is_set():
                with write_lock:
                    stats["failed"] += len(chunk)
                print(f"  chunk of {len(chunk)} failed ({e})", flush=True)

    # Process in chunks
    chunks = [todo[i:i + config.batch_size] for i in range(0, len(todo), config.batch_size)]
    
    if config.max_concurrent > 1:
        with ThreadPoolExecutor(max_workers=config.max_concurrent) as pool:
            list(pool.map(process_chunk, chunks))
    else:
        for chunk in chunks:
            process_chunk(chunk)

    out_file.close()

    # Print summary
    with open(out_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows:
        print(f"done: no rewrites cached ({stats['failed']} failed)")
        return

    lat = sorted(r["latency_s"] for r in rows)
    tok = [r.get("total_tokens", 0) for r in rows if r.get("total_tokens", 0) > 0]
    
    # Compare with original formal queries
    norm_func = lambda s: " ".join(s.lower().replace("?", " ").split())
    same = sum(1 for r in rows if r["query_id"] in tp.index 
               and norm_func(r["normalized"]) == norm_func(tp.at[r["query_id"], "formal_query"]))
    
    print(f"done: {stats['ok']} new, {stats['failed']} failed, {len(rows)} cached total | "
          f"latency mean {sum(lat)/len(lat):.2f}s p50 {lat[len(lat)//2]:.2f}s p95 {lat[int(0.95*len(lat))]:.2f}s | "
          f"tokens mean {sum(tok)/len(tok):.1f} | "
          f"identical to original formal query: {same}/{len(rows)} ({100*same/len(rows):.1f}%)")


if __name__ == "__main__":
    main()