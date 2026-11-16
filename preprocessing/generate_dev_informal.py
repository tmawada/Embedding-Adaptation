"""Informal versions of the official MIRACL-id dev queries (data/miracl_dev/queries.json) in two styles, with Gemini.

  template  imitates the informal side of data/train_pairs.csv, the style of every training query. The prompt that
            made train_pairs.csv is not available, so few-shot examples drawn from our train split define the style.
  varied    the style sampler of ../BGE_M3/preprocess/generate_informal.py: a style profile drawn per query (particle,
            abbreviation, slang, typos, code-mixing, word order, surface, length; seeded), with its system prompt and
            worked examples. Same generator design as the 644 style-sampled pairs of the style-shift pilot.

The generator (gemini-2.5-flash) is not the LLM-rewrite baseline's model (gemini-3.5-flash-lite), so the baseline is
not asked to undo its own rewrites. (A first attempt with gemini-3.5-flash stopped at the free tier's 20 requests a
day; its partial output is kept in data/miracl_dev/partial_gemini-3.5-flash/ and not used.) Results go to data/miracl_dev/informal_<style>.jsonl (one record per query,
resumable: a re-run only fills what is missing), then a marker audit compares each style with its reference file.

  python3 preprocessing/generate_dev_informal.py --style template
  python3 preprocessing/generate_dev_informal.py --style varied
  python3 preprocessing/generate_dev_informal.py --style varied --audit     # audit only, no API calls

Only the public MIRACL-id dev questions are sent to Google. The key is read like methods/llm_rewrite/normalize.py does.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from preprocessing.dataset import load_train_pairs  # noqa: E402
from methods.llm_rewrite.normalize import QuotaExhausted, RateLimiter, call_gemini, load_api_key  # noqa: E402

DEV = os.path.join(ROOT, "data", "miracl_dev")
GENERATOR = os.path.join(os.path.dirname(ROOT), "BGE_M3", "preprocess", "generate_informal.py")
STYLE_PAIRS = os.path.join(os.path.dirname(ROOT), "BGE_M3", "data", "pairs", "new_pairs.csv")
PROMPT_VERSION = {"template": "dev-template-v2", "varied": "dev-varied-v1"}

TEMPLATE_PROMPT = """\
You rewrite formal Indonesian search questions into informal Indonesian. The informal versions must be in exactly \
the style of the examples below: they are compared with a dataset written in that style, so copy its habits \
(casing, punctuation, word order, particles, abbreviations, length) as closely as you can.

Rules:
1. Keep the information need: someone reading only the rewrite must look for the same answer. Do not answer it.
2. Keep the spelling of every named entity, number and date (casing may follow the style).
3. Match the dataset's mix, not just its most common pattern: about 69% of its rewrites end in "sih", 16% in "ya", 7% in "nih", 3% in "dong", 3% in "ga"/"gak". It abbreviates sparingly ("yg" in only about 6% of rewrites, "brp" in 16%). Spread your batch the same way.
4. Write only the rewrite for each item.

Examples (formal -> informal):
"""

SCHEMA = {"type": "OBJECT", "required": ["rewrites"],
          "properties": {"rewrites": {"type": "ARRAY", "items": {
              "type": "OBJECT", "required": ["id", "informal"],
              "properties": {"id": {"type": "STRING"}, "informal": {"type": "STRING"}}}}}}


def load_generator():
    """../BGE_M3/preprocess/generate_informal.py as a module: its style axes, sampler, prompt and markers."""
    spec = importlib.util.spec_from_file_location("generate_informal", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # top-level imports are standard library only; anthropic is imported lazily
    return module


def template_prompt(n_shots: int, seed: int) -> str:
    """The template instruction plus n_shots (formal, informal) pairs from our train split."""
    with open(os.path.join(ROOT, "methods", "cache", "splits.json")) as f:
        train = json.load(f)["train"]
    tp = load_train_pairs(os.path.join(ROOT, "data")).set_index("query_id")
    shots = random.Random(seed).sample(train, n_shots)
    return TEMPLATE_PROMPT + "\n".join(f"{tp.at[q, 'formal_query']} -> {tp.at[q, 'generated_informal_query']}"
                                       for q in shots)


def batch_prompt(style: str, items: list, gen, header: str) -> str:
    """items: (query_id, formal, profile or None). Ends with the JSON instruction."""
    if style == "varied":
        body = gen.build_batch_prompt(items)
    else:
        lines = ["Rewrite these. Return one entry per item.", ""]
        for i, (qid, formal, _) in enumerate(items, 1):
            lines += [f"[{i}] id={qid}", f"    formal: {formal}", ""]
        body = "\n".join(lines)
    return (f"{header}\n\n{body}\nAnswer with JSON: {{\"rewrites\": [{{\"id\": <the id given>, "
            "\"informal\": <the rewrite>}}, ...]}, one entry per item.")


def audit(rows: list, reference: list, gen, name: str):
    """Marker shares of the generated informal queries next to a reference set of the same style."""
    def stats(texts):
        n = len(texts)
        out = {"n": n,
               "ends with 'sih'": sum(t.strip().endswith("sih") for t in texts) / n,
               "ends with '?'": sum(t.strip().endswith("?") for t in texts) / n,
               "starts with a capital": sum(t[:1].isupper() for t in texts) / n,
               "mean words": sum(len(t.split()) for t in texts) / n}
        for label, pattern in gen.MARKERS.items():
            out[label] = sum(bool(re.search(pattern, t.lower())) for t in texts) / n
        return out
    a, b = stats([r["informal"] for r in rows]), stats(reference)
    print(f"\naudit: {len(rows)} generated vs {name} ({b['n']})")
    for k in a:
        if k == "n":
            continue
        fmt = (lambda v: f"{v:6.1f}") if k == "mean words" else (lambda v: f"{100 * v:5.1f}%")
        print(f"  {k:32s} {fmt(a[k])}   {fmt(b[k])}")
    same = sum(" ".join(r["informal"].lower().split()) == " ".join(r["formal"].lower().split()) for r in rows)
    print(f"  identical to the formal query     {same}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True, choices=["template", "varied"])
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--api_key_file", default=None)
    ap.add_argument("--batch_size", type=int, default=45, help="Queries per request (the free tier caps requests)")
    ap.add_argument("--max_output_tokens", type=int, default=32768, help="Thinking tokens count against it")
    ap.add_argument("--thinking_budget", type=int, default=8192,
                    help="Cap on thinking tokens per request: uncapped, gemini-2.5-flash thought ~700 tokens per varied "
                         "item and a 55-item batch ran out of output budget mid-JSON")
    ap.add_argument("--shots", type=int, default=40, help="template: few-shot pairs from our train split")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--rpm", type=float, default=4)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42, help="Style profiles (varied) and few-shot sample (template)")
    ap.add_argument("--audit", action="store_true", help="Only audit the existing output, no API calls")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N dev queries (smoke test)")
    args = ap.parse_args()
    gen = load_generator()
    out_path = os.path.join(DEV, f"informal_{args.style}.jsonl")

    with open(os.path.join(DEV, "queries.json")) as f:
        queries = [(r["query_id"], r["query"]) for r in json.load(f)][: args.limit]
    # Profiles drawn in query order from one seeded RNG, as generate_informal.generate() does.
    rng = random.Random(args.seed)
    profiles = {q: gen.sample_style(rng) for q, _ in queries} if args.style == "varied" else {}
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {r["query_id"]: r for r in map(json.loads, filter(str.strip, f))}
    todo = [(q, text, profiles.get(q)) for q, text in queries if q not in done]
    print(f"[{args.style}] {len(queries)} dev queries, {len(done)} done, {len(todo)} to generate "
          f"with {args.model} -> {os.path.relpath(out_path, ROOT)}")

    if todo and not args.audit:
        header = (f"{gen.SYSTEM_PROMPT}\n\n{gen.style_legend()}\n{gen.FEW_SHOT}" if args.style == "varied"
                  else template_prompt(args.shots, args.seed))
        key, limiter, lock = load_api_key(args.api_key_file), RateLimiter(args.rpm), threading.Lock()
        out_file, stop = open(out_path, "a", encoding="utf-8"), threading.Event()
        config = {"temperature": args.temperature, "responseMimeType": "application/json", "responseSchema": SCHEMA,
                  "thinkingConfig": {"thinkingBudget": args.thinking_budget}}

        def run(batch):
            if stop.is_set():
                return
            try:
                text, latency, attempts = call_gemini(key, args.model, batch_prompt(args.style, batch, gen, header),
                                                      limiter, max_output_tokens=args.max_output_tokens,
                                                      generation_config=config)
                got = {str(e["id"]).strip(): gen._clean(str(e["informal"])) for e in json.loads(text)["rewrites"]
                       if str(e.get("informal", "")).strip()}
            except QuotaExhausted as e:
                stop.set()
                print(f"  QUOTA EXHAUSTED, stopping (re-run later to resume): {e}", flush=True)
                return
            except Exception as e:
                print(f"  batch of {len(batch)} failed ({e.__class__.__name__}: {str(e)[:200]}); re-run to retry",
                      flush=True)
                return
            with lock:
                for qid, formal, profile in batch:
                    if qid not in got:
                        continue
                    rec = {"query_id": qid, "formal": formal, "informal": got[qid],
                           "style": gen.style_key(profile) if profile else "template", "model": args.model,
                           "prompt_version": PROMPT_VERSION[args.style], "temperature": args.temperature,
                           "thinking_budget": args.thinking_budget,
                           "latency_s": round(latency / len(batch), 3), "attempts": attempts}
                    out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[qid] = rec
                out_file.flush()
                missing = sum(q not in got for q, _, _ in batch)
                print(f"  {len(done)}/{len(queries)} done" + (f" ({missing} missing from this batch)" if missing
                                                              else ""), flush=True)

        batches = [todo[i: i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(run, batches))
        out_file.close()

    rows = [done[q] for q, _ in queries if q in done]
    if len(rows) < len(queries):
        print(f"[{args.style}] {len(queries) - len(rows)} queries still missing: re-run to fill them")
    if rows:
        if args.style == "template":
            reference = load_train_pairs(os.path.join(ROOT, "data")).generated_informal_query.tolist()
            audit(rows, reference, gen, "data/train_pairs.csv (all 4,071)")
        else:
            reference = pd.read_csv(STYLE_PAIRS, dtype=str).informal.tolist()
            audit(rows, reference, gen, "../BGE_M3/data/pairs/new_pairs.csv")


if __name__ == "__main__":
    main()
