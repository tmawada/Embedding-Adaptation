"""Normalise informal Indonesian queries into formal ones with a Google AI Studio (Gemini) model.

The LLM-rewrite baseline: the step Query-DANN / query distillation are meant to replace.
Results are cached as JSONL (one record per query, resumable) together with per-call latency.

  export GEMINI_API_KEY=...            # or: --api_key_file ~/.gemini_api_key
  python3 llm_rewrite/normalize.py --list_models
  python3 llm_rewrite/normalize.py --split test --model gemini-2.5-flash
  python3 llm_rewrite/normalize.py --split test --batch_size 10 --rpm 15    # fewer calls if rate-limited

Only the informal query text is sent to Google (public MIRACL-id queries). The API key is read from
the environment or a file, never logged.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset import load_train_pairs  # noqa: E402

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
PROMPT_VERSION = "id-normalise-v1"

INSTRUCTION = """Tugas: ubah pertanyaan informal (bahasa gaul/sehari-hari) menjadi pertanyaan formal bahasa Indonesia baku untuk sistem pencarian dokumen.

Aturan:
- JANGAN menjawab pertanyaannya.
- Jangan menambah atau menghilangkan informasi.
- Pertahankan nama entitas, angka, dan ejaan nama asing apa adanya.
- Keluarkan HANYA pertanyaan formalnya, satu baris, tanpa penjelasan."""


def load_api_key(path: str = None) -> str:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_KEY"):
        key = os.environ.get(var, "").strip()
        if key:
            return key
    for candidate in filter(None, [path, os.path.expanduser("~/.gemini_api_key")]):
        if os.path.exists(candidate):
            with open(candidate) as f:
                key = f.read().strip()
            if key:
                return key
    raise SystemExit("No API key found. Set GEMINI_API_KEY or put the key in ~/.gemini_api_key "
                     "(chmod 600), or pass --api_key_file.")


class QuotaExhausted(RuntimeError):
    """The API returned 429 after the allowed retries: stop instead of burning more quota."""


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


def call_gemini(key: str, model: str, prompt: str, limiter: RateLimiter, max_output_tokens: int,
                max_retries: int = 4, no_thinking: bool = False, quota_attempts: int = 2):
    """Returns (text, latency_seconds, attempts). Raises RuntimeError after max_retries."""
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_output_tokens}}
    if no_thinking:
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
    url = f"{ENDPOINT}/{model}:generateContent"
    last = ""
    for attempt in range(1, max_retries + 1):
        limiter.wait()
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.load(r)
            latency = time.monotonic() - t0
            cand = (payload.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                last = f"empty response (finishReason={cand.get('finishReason')})"
                raise urllib.error.HTTPError(url, 502, last, None, None)
            return text, latency, attempt
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()[:500]
            except Exception:
                pass
            last = f"HTTP {e.code} {detail}"
            if e.code == 400 and no_thinking and "thinkingConfig" in body["generationConfig"]:
                # Some models reject thinkingBudget:0 with a generic INVALID_ARGUMENT; drop it and retry.
                body["generationConfig"].pop("thinkingConfig", None)
                no_thinking = False
                continue
            if e.code == 429 and attempt >= quota_attempts:
                # Every retry counts against the daily free-tier quota, so give up early.
                raise QuotaExhausted(last)
            if e.code in (400, 401, 403, 404) and "thinkingConfig" not in detail:
                raise RuntimeError(last)  # not retryable (bad key/model/request)
        except Exception as e:
            last = f"{e.__class__.__name__}: {e}"
        time.sleep(min(60, 2 ** attempt) + random.random())
    raise RuntimeError(f"failed after {max_retries} attempts: {last}")


def list_models(key: str):
    req = urllib.request.Request(ENDPOINT, headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    rows = [(m["name"].split("/")[-1], m.get("inputTokenLimit"), ",".join(m.get("supportedGenerationMethods", [])))
            for m in data.get("models", [])]
    print(f"{'model':<42} {'input tokens':>12}  methods")
    for name, lim, methods in sorted(rows):
        if "generateContent" in methods:
            print(f"{name:<42} {str(lim):>12}  {methods}")


def build_prompt(informal: str, shots: list) -> str:
    lines = [INSTRUCTION, ""]
    for inf, form in shots:
        lines += [f"Informal: {inf}", f"Formal: {form}", ""]
    lines += [f"Informal: {informal}", "Formal:"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--queries", default="informal", choices=["informal", "formal"],
                    help="Which query text to normalise: the informal queries, or the formal ones "
                         "(a deployed rewriter cannot know the input is already formal)")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--api_key_file", default=None)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"), help="For splits.json")
    ap.add_argument("--out", default=None, help="Default: llm_rewrite/cache/<model>_<split>.jsonl")
    ap.add_argument("--shots", type=int, default=3, help="Few-shot examples taken from the TRAIN split only")
    ap.add_argument("--batch_size", type=int, default=1, help=">1 normalises several queries per call (fewer requests)")
    ap.add_argument("--rpm", type=float, default=15, help="Requests per minute cap (free tier is limited)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max_retries", type=int, default=4, help="Retries per request for transient errors (not 429)")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N queries (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list_models", action="store_true", help="List models this key can use, then exit")
    args = ap.parse_args()

    key = load_api_key(args.api_key_file)
    if args.list_models:
        list_models(key)
        return
    tp = load_train_pairs(args.data_dir).set_index("query_id")
    column = {"informal": "generated_informal_query", "formal": "formal_query"}[args.queries]
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        splits = json.load(f)
    qids = splits[args.split][: args.limit]
    shots = [(tp.at[q, "generated_informal_query"], tp.at[q, "formal_query"])
             for q in random.Random(args.seed).sample(splits["train"], args.shots)]

    suffix = "" if args.queries == "informal" else "_formal"
    out_path = args.out or os.path.join(HERE, "cache", f"{args.model}_{args.split}{suffix}.jsonl")
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
    stop = threading.Event()  # set when the API quota is exhausted: stop instead of burning more of it

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
            text, latency, attempts = call_gemini(key, args.model, build_prompt(tp.at[qid, column], shots),
                                                  limiter, max_output_tokens=128, max_retries=args.max_retries)
            record(qid, text.splitlines()[0].strip(), latency, attempts, 1)
        except QuotaExhausted as e:
            if not stop.is_set():
                stop.set()
                print(f"  QUOTA EXHAUSTED, stopping (cached rewrites are kept; re-run later to resume): {e}", flush=True)
        except Exception as e:
            with write_lock:
                stats["failed"] += 1
            print(f"  FAILED {qid}: {e}", flush=True)

    def do_batch(chunk):
        if stop.is_set():
            return
        numbered = "\n".join(f"{i + 1}. {tp.at[q, column]}" for i, q in enumerate(chunk))
        prompt = (INSTRUCTION + "\n\nUbah setiap pertanyaan berikut. Keluarkan daftar bernomor dengan jumlah baris "
                  f"yang sama ({len(chunk)} baris), format `nomor. pertanyaan formal`, tanpa teks lain.\n\n" + numbered)
        try:
            text, latency, attempts = call_gemini(key, args.model, prompt, limiter, max_output_tokens=64 * len(chunk) + 128)
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
                      f"(pass --batch_fallback to retry these one by one, which costs one request each)", flush=True)
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
