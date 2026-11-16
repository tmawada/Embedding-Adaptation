#!/usr/bin/env python3
"""Normalise informal Indonesian queries into formal ones using local Ollama model (Qwen2.5 3B)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from preprocessing.dataset import load_train_pairs  # noqa: E402

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
PROMPT_VERSION = "id-normalise-v1-ollama"

INSTRUCTION = """Tugas: ubah pertanyaan informal (bahasa gaul/sehari-hari) menjadi pertanyaan formal bahasa Indonesia baku untuk sistem pencarian dokumen.

Aturan:
- JANGAN menjawab pertanyaannya.
- Jangan menambah atau menghilangkan informasi.
- Pertahankan nama entitas, angka, dan ejaan nama asing apa adanya.
- Keluarkan HANYA pertanyaan formalnya, satu baris, tanpa penjelasan."""


class RateLimiter:
    """At most `rpm` requests per minute across all threads."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.interval
        if start > now:
            time.sleep(start - now)


def call_ollama(model: str, prompt: str, limiter: RateLimiter, max_retries: int = 3) -> tuple[str, float, int]:
    """Returns (text, latency_seconds, attempts). Raises RuntimeError after max_retries."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 128,
        }
    }
    last = ""
    for attempt in range(1, max_retries + 1):
        limiter.wait()
        req = urllib.request.Request(
            OLLAMA_ENDPOINT,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json"}
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.load(r)
            latency = time.monotonic() - t0
            text = payload.get("response", "").strip()
            if not text:
                last = "empty response"
                raise RuntimeError(last)
            return text, latency, attempt
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()[:500]
            except Exception:
                pass
            last = f"HTTP {e.code} {detail}"
            if e.code in (400, 401, 403, 404):
                raise RuntimeError(last)
        except Exception as e:
            last = f"{e.__class__.__name__}: {e}"
        time.sleep(min(60, 2 ** attempt) + random.random())
    raise RuntimeError(f"failed after {max_retries} attempts: {last}")


def build_prompt(informal: str, shots: list) -> str:
    lines = [INSTRUCTION, ""]
    for inf, form in shots:
        lines += [f"Informal: {inf}", f"Formal: {form}", ""]
    lines += [f"Informal: {informal}", "Formal:"]
    return "\n".join(lines)


def build_batch_prompt(queries: list, shots: list) -> str:
    """Build prompt for batch processing multiple queries at once."""
    lines = [INSTRUCTION, ""]
    for inf, form in shots:
        lines += [f"Informal: {inf}", f"Formal: {form}", ""]
    lines += [f"Ubah setiap pertanyaan berikut. Keluarkan daftar bernomor dengan jumlah baris yang sama ({len(queries)} baris), format `nomor. pertanyaan formal`, tanpa teks lain.", ""]
    for i, q in enumerate(queries):
        lines.append(f"{i+1}. {q}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--queries", default="informal", choices=["informal", "formal", "informal_varied"],
                    help="Which query text to normalise")
    ap.add_argument("--model", default="qwen2.5:3b-instruct-q4_K_M", help="Ollama model name")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "methods", "cache"), help="For splits.json")
    ap.add_argument("--out", default=None, help="Default: llm_rewrite/cache/<model>_<split>.jsonl")
    ap.add_argument("--shots", type=int, default=3, help="Few-shot examples taken from the TRAIN split only")
    ap.add_argument("--batch_size", type=int, default=1, help=">1 normalises several queries per call")
    ap.add_argument("--rpm", type=float, default=30, help="Requests per minute cap")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max_retries", type=int, default=3, help="Retries per request for transient errors")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N queries (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_fallback", action="store_true",
                    help="Retry a failed batch one query at a time instead of skipping it")
    args = ap.parse_args()

    tp = load_train_pairs(args.data_dir).set_index("query_id")
    column = {"informal": "generated_informal_query", "formal": "formal_query",
              "informal_varied": "informal_varied"}[args.queries]
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        splits = json.load(f)
    qids = splits[args.split][: args.limit]
    shots = [(tp.at[q, "generated_informal_query"], tp.at[q, "formal_query"])
             for q in random.Random(args.seed).sample(splits["train"], args.shots)]

    # Sanitize model name for filename
    model_safe = args.model.replace(":", "_").replace("/", "_").replace("-", "_")
    suffix = {"informal": "", "formal": "_formal", "informal_varied": "_varied"}[args.queries]
    out_path = args.out or os.path.join(HERE, "cache", f"{model_safe}_{args.split}{suffix}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(l)["query_id"]: 1 for l in f if l.strip()}
    todo = [q for q in qids if q not in done]
    print(f"[{args.model}] split={args.split} queries={len(qids)} cached={len(done)} to do={len(todo)} "
          f"batch_size={args.batch_size} rpm={args.rpm} shots={args.shots} -> {os.path.relpath(out_path, ROOT)}")
    if not todo:
        return

    limiter = RateLimiter(args.rpm)
    write_lock = threading.Lock()
    out_file = open(out_path, "a")
    stats = {"ok": 0, "failed": 0}
    stop = threading.Event()

    def record(qid, text, latency, attempts, batch_size):
        with write_lock:
            out_file.write(json.dumps({"query_id": qid, "queries": args.queries, "informal": tp.at[qid, column],
                                       "normalized": text, "latency_s": round(latency, 3), "attempts": attempts,
                                       "model": args.model, "prompt_version": PROMPT_VERSION,
                                       "shots": args.shots, "batch_size": batch_size}, ensure_ascii=False) + "\n")
            out_file.flush()
            stats["ok"] += 1
            n = stats["ok"] + stats["failed"]
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} done (failed {stats['failed']})", flush=True)

    def do_single(qid):
        if stop.is_set():
            return
        try:
            text, latency, attempts = call_ollama(args.model, build_prompt(tp.at[qid, column], shots),
                                                  limiter, max_retries=args.max_retries)
            record(qid, text.splitlines()[0].strip(), latency, attempts, 1)
        except Exception as e:
            with write_lock:
                stats["failed"] += 1
            print(f"  FAILED {qid}: {e}", flush=True)

    def do_batch(chunk):
        if stop.is_set():
            return
        queries = [tp.at[q, column] for q in chunk]
        try:
            text, latency, attempts = call_ollama(args.model, build_batch_prompt(queries, shots),
                                                  limiter, max_retries=args.max_retries)
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            parsed = []
            for l in lines:
                head, _, rest = l.partition(".")
                if head.strip().isdigit() and rest.strip():
                    parsed.append(rest.strip())
            if len(parsed) != len(chunk):
                raise RuntimeError(f"expected {len(chunk)} lines, parsed {len(parsed)}")
            for qid, norm in zip(chunk, parsed):
                record(qid, norm, latency / len(chunk), attempts, len(chunk))
        except Exception as e:
            if not args.batch_fallback:
                with write_lock:
                    stats["failed"] += len(chunk)
                print(f"  batch of {len(chunk)} failed ({e}); skipped "
                      f"(pass --batch_fallback to retry these one by one)", flush=True)
                return
            print(f"  batch failed ({e}); falling back to per-query", flush=True)
            for qid in chunk:
                do_single(qid)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        if args.batch_size > 1:
            chunks = [todo[i: i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
            list(pool.map(do_batch, chunks))
        else:
            list(pool.map(do_single, todo))
    out_file.close()

    with open(out_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows:
        print(f"done: no rewrites cached ({stats['failed']} failed). Nothing written to {os.path.relpath(out_path, ROOT)}")
        return
    lat = sorted(r["latency_s"] for r in rows)
    same = sum(1 for r in rows if r["normalized"].strip().lower().rstrip("?").strip()
               == tp.at[r["query_id"], "formal_query"].strip().lower().rstrip("?").strip())
    print(f"done: {stats['ok']} new, {stats['failed']} failed, {len(rows)} cached total | "
          f"latency mean {sum(lat)/len(lat):.2f}s p50 {lat[len(lat)//2]:.2f}s p95 {lat[int(0.95*len(lat))]:.2f}s | "
          f"exactly equal to the original formal query: {same}/{len(rows)} ({100*same/len(rows):.1f}%)")


if __name__ == "__main__":
    main()