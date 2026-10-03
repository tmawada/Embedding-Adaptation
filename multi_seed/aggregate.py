"""Mean ± standard deviation across training seeds for every trained model (test split).

For each model and seed: Recall@100, MRR@100, nDCG@10 on formal and informal queries, then
  1. main table      each cell = mean ± SD over seeds (sample SD, ddof = 1)
  2. nDCG@10 gap     formal, informal, gap, narrowing vs base — each mean ± SD
  3. comparisons     difference mean ± SD across seeds + paired bootstrap on seed-averaged per-query scores
Seed 42 reuses the existing checkpoints; seeds 43-46 come from multi_seed/run.sh.
BGE-M3 base and LLM normalisation involve no random training: one deterministic run, SD = 0.

  python3 multi_seed/aggregate.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "query_distillation"))
sys.path.insert(0, os.path.join(ROOT, "comparison"))

from dataset import EmbeddingStore, load_qrels  # noqa: E402
from eval import dense_search, paired_bootstrap  # noqa: E402
from evaluate import adapt, load_adapter  # noqa: E402  (query_distillation/evaluate.py)
from model import BGEM3Encoder  # noqa: E402
from report import per_query  # noqa: E402  (comparison/report.py: Recall@100, MRR@100, nDCG@10)

H = "comparison/runs/harmonized"
A = f"{H}/ablation_adapter"
METRICS = ("Recall@100", "MRR@100", "nDCG@10")
SEEDS = (42, 43, 44, 45, 46)

# (key, table label, tower, seed-42 checkpoint)
MODELS = [
    ("dann_basic", "Query-DANN v2 (basic adapter)", "base", f"{H}/base/query_dann_v2/g1.0/best"),
    ("dann_linear", "Query-DANN v2 (linear adapter)", "base", f"{A}/dann/linear/best"),
    ("distill_score_basic", "Query distillation score (basic adapter)", "base", f"{H}/base/query_distillation/score/best"),
    ("distill_score_linear", "Query distillation score (linear adapter)", "base", f"{A}/distill/linear/best"),
    ("distill_embed_basic", "Query distillation embed (basic adapter)", "base", f"{H}/base/query_distillation/embed/best"),
    ("distill_embed_linear", "Query distillation embed (linear adapter)", "base", f"{A}/distill/linear_embed/best"),
    ("tsdae_dann_linear", "TSDAE-SAPT (task queries) + Query-DANN v2 (linear adapter)", "tsdae", f"{A}/sapt_queries/dann_linear/best"),
    ("tsdae_distill_score_linear", "TSDAE-SAPT (task queries) + Query distillation score (linear adapter)", "tsdae", f"{A}/sapt_queries/distill_linear/best"),
    ("stack_tsdae_dann_kd", "TSDAE-SAPT (task queries) + Query-DANN v2 + distillation score (linear adapter)", "tsdae", "stacked/runs/sapt_queries/kd1.0/best"),
]
COMPARISONS = [
    ("dann_linear", "dann_basic"),
    ("dann_linear", "distill_score_linear"),
    ("distill_score_linear", "distill_embed_linear"),
    ("tsdae_dann_linear", "dann_linear"),
    ("stack_tsdae_dann_kd", "dann_linear"),
]


def root(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def ckpt(key, seed42, seed):
    return root(seed42) if seed == 42 else os.path.join(HERE, "runs", key, f"seed{seed}", "best")


def fmt(values, digits=4, scale=1.0, signed=False):
    v = np.asarray(values, dtype=float) * scale
    sign = "+" if signed else ""
    if len(v) < 2:
        return f"{v.mean():{sign}.{digits}f}"
    return f"{v.mean():{sign}.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites", default="llm_rewrite/cache/gemini-3.5-flash-lite_test.jsonl")
    ap.add_argument("--tsdae_cache", default=f"{H}/sapt_queries/cache")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "runs"))
    args = ap.parse_args()

    store = EmbeddingStore(os.path.join(ROOT, "cache"))
    device = store.device
    with open(os.path.join(ROOT, "cache", "splits.json")) as f:
        test = json.load(f)["test"]
    in_corpus = set(store.doc2row)
    full = load_qrels(os.path.join(ROOT, "data"))
    qrels = {q: full[q] & in_corpus for q in test if full.get(q, set()) & in_corpus}
    qids = sorted(qrels, key=int)
    rows = torch.tensor([store.qid2row[q] for q in qids], device=device)
    towers = {"base": {"formal": store.formal[rows].float(), "informal": store.informal[rows].float()}}
    with open(os.path.join(root(args.tsdae_cache), "query_ids.json")) as f:
        order = {q: i for i, q in enumerate(json.load(f))}
    sel = [order[q] for q in qids]
    towers["tsdae"] = {k: F.normalize(torch.from_numpy(np.asarray(
        np.load(os.path.join(root(args.tsdae_cache), f"{k}_emb.npy"), mmap_mode="r")[sel], dtype=np.float32)).to(device), dim=-1)
        for k in ("formal", "informal")}

    def evaluate(z):
        _, idx = dense_search(z, store.corpus, 100)
        return per_query(idx.tolist(), qids, qrels, store.doc_ids)

    # results[label] = list over seeds of {"formal": per-query metrics, "informal": ...}
    results, seeds_used = {}, {}
    results["BGE-M3 base"] = [{k: evaluate(v) for k, v in towers["base"].items()}]
    seeds_used["BGE-M3 base"] = ["deterministic"]
    with open(root(args.rewrites)) as f:
        rw = {r["query_id"]: r["normalized"] for r in map(json.loads, filter(str.strip, f))}
    enc = BGEM3Encoder()
    z_llm = torch.from_numpy(enc.encode([rw[q] for q in qids], max_length=64, progress=False)).to(device).float()
    del enc
    torch.cuda.empty_cache()
    results["BGE-M3 + LLM normalisation"] = [{"formal": results["BGE-M3 base"][0]["formal"], "informal": evaluate(z_llm)}]
    seeds_used["BGE-M3 + LLM normalisation"] = ["deterministic"]

    key2label = {}
    for key, label, tower, seed42 in MODELS:
        key2label[key] = label
        runs, used = [], []
        for seed in SEEDS:
            path = ckpt(key, seed42, seed)
            if not os.path.exists(os.path.join(path, "adapter.pt")):
                print(f"! {label}: missing seed {seed} ({os.path.relpath(path, ROOT)})")
                continue
            a = load_adapter(path, device)
            runs.append({k: evaluate(adapt(a, v)) for k, v in towers[tower].items()})
            used.append(seed)
        results[label], seeds_used[label] = runs, used
        n_inf = [r["informal"]["nDCG@10"].mean() for r in runs]
        print(f"{label:<82} seeds {used} informal nDCG@10 {fmt(n_inf)}")

    base_gap = (results["BGE-M3 base"][0]["informal"]["nDCG@10"].mean()
                - results["BGE-M3 base"][0]["formal"]["nDCG@10"].mean())
    n_max = max(len(v) for v in seeds_used.values() if v and v[0] != "deterministic")
    lines = [f"# Mean ± SD over training seeds — MIRACL-id test ({len(qids)} queries, {len(store.doc_ids):,} passages)", "",
             f"BGE-M3 dense retrieval. Trained models: mean ± sample SD over seeds {list(SEEDS)} "
             f"(n per row in the JSON; harmonized protocol, gamma = 1.0, kd_weight = 1.0). "
             "The TSDAE (task queries) tower is fixed at its seed-42 checkpoint; only the adapters vary. "
             "BGE-M3 base and LLM normalisation are deterministic (single run, no SD).", "",
             "| Model | Formal R@100 | Formal MRR@100 | Formal nDCG@10 | Informal R@100 | Informal MRR@100 | Informal nDCG@10 |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for label, runs in results.items():
        cells = []
        summary[label] = {"seeds": seeds_used[label]}
        for side in ("formal", "informal"):
            for m in METRICS:
                vals = [r[side][m].mean() for r in runs]
                cells.append(fmt(vals))
                summary[label][f"{side}_{m}"] = {"per_seed": [float(x) for x in vals], "mean": float(np.mean(vals)),
                                                 "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "## nDCG@10 formal/informal gap (mean ± SD over seeds)", "",
              f"Gap = informal − formal in nDCG@10 points. Narrowing vs base = gap − base gap ({100 * base_gap:.2f}).", "",
              "| Model | Formal nDCG@10 | Informal nDCG@10 | Gap (pts) | Narrowing vs base (pts) |", "|---|---|---|---|---|"]
    for label, runs in results.items():
        f_ = [r["formal"]["nDCG@10"].mean() for r in runs]
        i_ = [r["informal"]["nDCG@10"].mean() for r in runs]
        g = [b - a for a, b in zip(f_, i_)]
        lines.append(f"| {label} | {fmt(f_)} | {fmt(i_)} | {fmt(g, 2, 100, True)} | {fmt([x - base_gap for x in g], 2, 100, True)} |")
        summary[label]["gap_pts"] = [100 * x for x in g]

    lines += ["", "## Key comparisons (nDCG@10 points)", "",
              "Across seeds = per-seed difference, mean ± SD. p = paired bootstrap on seed-averaged per-query scores.", "",
              "| A | B | Informal A − B (seeds) | p | Formal A − B (seeds) | p |", "|---|---|---|---|---|---|"]
    tests = {}
    for ka, kb in COMPARISONS:
        la, lb = key2label[ka], key2label[kb]
        ra, rb = results[la], results[lb]
        cells = []
        for side in ("informal", "formal"):
            common = min(len(ra), len(rb))
            diffs = [ra[i][side]["nDCG@10"].mean() - rb[i][side]["nDCG@10"].mean() for i in range(common)]
            avg_a = np.mean([r[side]["nDCG@10"] for r in ra], axis=0)
            avg_b = np.mean([r[side]["nDCG@10"] for r in rb], axis=0)
            bt = paired_bootstrap(avg_a, avg_b, args.bootstrap)
            tests[f"{la} vs {lb} ({side})"] = {"per_seed_diff": [float(d) for d in diffs], "bootstrap": bt}
            cells += [fmt(diffs, 2, 100, True), f"{bt['p']:.3f}{'*' if bt['p'] < 0.05 else ''}"]
        lines.append(f"| {la} | {lb} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "seed_report_test.md"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out_dir, "seed_report_test.json"), "w") as f:
        json.dump({"n_queries": len(qids), "seeds": list(SEEDS), "models": summary, "comparisons": tests}, f, indent=2)
    print("\n" + text)


if __name__ == "__main__":
    main()
