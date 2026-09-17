"""Final report: one row per model, formal and informal columns, plus the nDCG@10 formal/informal gap.

Metrics are computed here (not via eval.py) because this report uses MRR@100:
  Recall@100  share of a query's relevant passages found in the top 100
  MRR@100     1 / rank of the first relevant passage within the top 100
  nDCG@10     binary-relevance nDCG at 10

Every row is BGE-M3 dense retrieval over the same frozen 500k index and the same held-out queries; each
row is a full query pipeline, evaluated twice: once given formal queries, once given informal queries.

  python3 comparison/report.py                      # harmonised runs (default)
  python3 comparison/report.py --tuned              # the earlier, per-method-tuned checkpoints
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "query_distillation"))

from dataset import EmbeddingStore, load_qrels  # noqa: E402
from eval import dense_search, paired_bootstrap  # noqa: E402
from evaluate import adapt, load_adapter  # noqa: E402
from model import BGEM3Encoder  # noqa: E402

DEPTH = 100


def abspath(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def per_query(rows, qids, qrels, doc_ids):
    """Recall@100, MRR@100, nDCG@10 per query, as parallel arrays."""
    rec, mrr, ndcg = [], [], []
    for qid, row in zip(qids, rows):
        rel = qrels[qid]
        hits = [doc_ids[j] in rel for j in row]
        rec.append(sum(hits) / len(rel))
        mrr.append(next((1.0 / (r + 1) for r, h in enumerate(hits) if h), 0.0))
        dcg = sum(1.0 / math.log2(r + 2) for r, h in enumerate(hits[:10]) if h)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(rel), 10)))
        ndcg.append(dcg / idcg)
    return {"Recall@100": np.array(rec), "MRR@100": np.array(mrr), "nDCG@10": np.array(ndcg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--tuned", action="store_true", help="Use the per-method-tuned checkpoints instead of harmonised")
    ap.add_argument("--rewrites_informal", default="llm_rewrite/cache/gemini-3.5-flash-lite_test.jsonl")
    ap.add_argument("--rewrites_formal", default="llm_rewrite/cache/gemini-3.5-flash-lite_test_formal.jsonl")
    ap.add_argument("--sapt_variants", nargs="*", default=["reviews", "queries"], choices=["reviews", "queries"],
                    help="Which SAPT towers to include: 'reviews' is the method (external informal corpus), "
                         "'queries' is the data ablation")
    ap.add_argument("--skip_default_adapters", action="store_true",
                    help="Omit the default (bottleneck) Query-DANN / distillation rows on the base tower")
    ap.add_argument("--skip_sapt_stacks", action="store_true",
                    help="Keep SAPT tower rows but omit their default (bottleneck) Query-DANN stack rows")
    ap.add_argument("--report_name", default=None, help="Output file tag (default: harmonized / tuned)")
    ap.add_argument("--extra_tower", action="append", default=[],
                    help="Extra rows on a chosen tower: 'Row name=base|reviews|queries=path/to/best' (repeatable)")
    ap.add_argument("--extra", action="append", default=[],
                    help="Extra adapter rows on the base tower: 'Row name=path/to/best' (repeatable)")
    ap.add_argument("--llm_formal", default="original", choices=["original", "rewrite"],
                    help="LLM normalisation row, formal column: 'original' = formal queries go to BGE-M3 unchanged "
                         "(normalisation applies to informal queries only); 'rewrite' = formal queries are also rewritten")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "runs", "harmonized"))
    args = ap.parse_args()

    H = os.path.join(HERE, "runs", "harmonized")
    if args.tuned:
        towers = {"reviews": "tsdae_sapt/cache/sapt_query_only",
                  "queries": "tsdae_sapt_queries/runs/tsdae_informal/cache"}
        ckpt = {"dann": "runs/v2_dann_g1.0/best", "score": "query_distillation/runs/score/best",
                "embed": "query_distillation/runs/embed/best",
                "reviews_dann": "encoder_comparison/runs/sapt_reviews_bge_m3/query_dann_v2",
                "queries_dann": "encoder_comparison/runs/tsdae_bge_m3/query_dann_v2"}
    else:
        towers = {"reviews": f"{H}/sapt_reviews/cache", "queries": f"{H}/sapt_queries/cache"}
        ckpt = {"dann": f"{H}/base/query_dann_v2", "score": f"{H}/base/query_distillation/score/best",
                "embed": f"{H}/base/query_distillation/embed/best",
                "reviews_dann": f"{H}/sapt_reviews/query_dann_v2", "queries_dann": f"{H}/sapt_queries/query_dann_v2"}

    def best_dann(path):
        path = abspath(path)
        if os.path.exists(os.path.join(path, "adapter.pt")):
            return path
        if not os.path.isdir(path):
            return None
        c = [os.path.join(path, d, "best") for d in os.listdir(path) if d != "g0"]
        c = [x for x in c if os.path.exists(os.path.join(x, "config.json"))]
        return max(c, key=lambda x: json.load(open(os.path.join(x, "config.json")))["dev"]["nDCG@10"]) if c else None

    store = EmbeddingStore(args.cache_dir)
    device = store.device
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        split_qids = json.load(f)[args.split]
    in_corpus = set(store.doc2row)
    full = load_qrels(args.data_dir)
    qrels = {q: full[q] & in_corpus for q in split_qids if full.get(q, set()) & in_corpus}
    qids = sorted(qrels, key=int)
    rows_idx = torch.tensor([store.qid2row[q] for q in qids], device=device)
    base = {"formal": store.formal[rows_idx].float(), "informal": store.informal[rows_idx].float()}

    def evaluate(z):
        _, idx = dense_search(z, store.corpus, DEPTH)
        return per_query(idx.tolist(), qids, qrels, store.doc_ids)

    def tower_vectors(cache):
        cache = abspath(cache)
        if not os.path.exists(os.path.join(cache, "informal_emb.npy")):
            return None
        with open(os.path.join(cache, "query_ids.json")) as f:
            order = {q: i for i, q in enumerate(json.load(f))}
        sel = [order[q] for q in qids]
        out = {}
        for kind in ("formal", "informal"):
            emb = np.load(os.path.join(cache, f"{kind}_emb.npy"), mmap_mode="r")
            out[kind] = F.normalize(torch.from_numpy(np.asarray(emb[sel], dtype=np.float32)).to(device), dim=-1)
        return out

    systems = {}  # name -> {"formal": metrics, "informal": metrics}

    def add(name, vectors):
        if vectors is None:
            print(f"! skipping {name}")
            return
        systems[name] = {k: evaluate(v) for k, v in vectors.items()}
        m = systems[name]
        print(f"{name:<44} formal nDCG {m['formal']['nDCG@10'].mean():.4f} | informal nDCG {m['informal']['nDCG@10'].mean():.4f}")

    add("BGE-M3 base", base)

    # LLM normalisation: the rewriter runs on whatever query it is given.
    llm = {}
    enc = None
    for kind, path in (("informal", args.rewrites_informal), ("formal", args.rewrites_formal)):
        if kind == "formal" and args.llm_formal == "original":
            continue  # formal queries are not rewritten
        p = abspath(path)
        if not os.path.exists(p):
            print(f"! no {kind} rewrites at {path}")
            continue
        with open(p) as f:
            rw = {r["query_id"]: r["normalized"] for r in map(json.loads, filter(str.strip, f))}
        if [q for q in qids if q not in rw]:
            print(f"! incomplete {kind} rewrites")
            continue
        enc = enc or BGEM3Encoder()
        llm[kind] = torch.from_numpy(enc.encode([rw[q] for q in qids], max_length=128,
                                                desc=f"LLM-normalised {kind}")).to(device).float()
    if enc is not None:
        del enc
        torch.cuda.empty_cache()
    if args.llm_formal == "original" and "informal" in llm:
        llm["formal"] = base["formal"]  # formal column = the original formal query through base BGE-M3
    if len(llm) == 2:
        add("BGE-M3 + LLM normalisation", llm)
    elif llm:
        print("! LLM normalisation needs both formal and informal rewrites for the table")

    for name, key in ([] if args.skip_default_adapters else
                      (("Query-DANN v2", "dann"), ("Query distillation (score)", "score"),
                       ("Query distillation (embed)", "embed"))):
        c = best_dann(ckpt[key])
        if c:
            a = load_adapter(c, device)
            add(name, {k: adapt(a, v) for k, v in base.items()})
        else:
            print(f"! missing checkpoint for {name}")

    for spec in args.extra:
        name, path = spec.split("=", 1)
        c = best_dann(path)
        if c:
            a = load_adapter(c, device)
            add(name, {k: adapt(a, v) for k, v in base.items()})
        else:
            print(f"! missing checkpoint for {name} ({path})")

    variants = [("TSDAE-SAPT (reviews)", "reviews", "reviews_dann"),
                ("TSDAE-SAPT (task queries)", "queries", "queries_dann")]
    for label, tkey, dkey in [v for v in variants if v[1] in args.sapt_variants]:
        vec = tower_vectors(towers[tkey])
        add(label, vec)
        c = None if args.skip_sapt_stacks else best_dann(ckpt[dkey])
        if vec and c:
            a = load_adapter(c, device)
            add(f"{label} + Query-DANN v2", {k: adapt(a, v) for k, v in vec.items()})

    # extra rows attached to a specific tower (e.g. a pass-through adapter on a SAPT tower)
    if args.extra_tower:
        pool = {"base": base, "reviews": tower_vectors(towers["reviews"]), "queries": tower_vectors(towers["queries"])}
        for spec in args.extra_tower:
            name, tkey, path = spec.split("=", 2)
            vec, c = pool.get(tkey), best_dann(path)
            if vec is None or not c:
                print(f"! skipping {name} (tower={tkey}, ckpt={path})")
                continue
            a = load_adapter(c, device)
            add(name, {k: adapt(a, v) for k, v in vec.items()})

    order = ("Recall@100", "MRR@100", "nDCG@10")
    lines = [f"# Retrieval results — MIRACL-id {args.split} ({len(qids)} queries, {len(store.doc_ids):,} passages)",
             "", f"BGE-M3 dense retrieval, top-{DEPTH}. "
             f"{'Per-method tuned settings.' if args.tuned else 'Harmonised training protocol for every trained model.'}",
             "", "| Model | Formal R@100 | Formal MRR@100 | Formal nDCG@10 | Informal R@100 | Informal MRR@100 | Informal nDCG@10 |",
             "|---|---|---|---|---|---|---|"]
    for name, m in systems.items():
        cells = [f"{m[side][k].mean():.4f}" for side in ("formal", "informal") for k in order]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    base_gap = systems["BGE-M3 base"]["informal"]["nDCG@10"].mean() - systems["BGE-M3 base"]["formal"]["nDCG@10"].mean()
    lines += ["", "## nDCG@10 formal/informal gap", "",
              "Gap = informal − formal (negative = informal is worse). Narrowing vs base = how much of the "
              f"base model's {abs(100 * base_gap):.2f}-point gap this pipeline removes. p tests informal vs formal "
              f"within the model (paired bootstrap, {args.bootstrap} samples).", "",
              "| Model | Formal nDCG@10 | Informal nDCG@10 | Gap (pts) | Gap (%) | Narrowing vs base | p |",
              "|---|---|---|---|---|---|---|"]
    gaps = {}
    for name, m in systems.items():
        f_m, i_m = m["formal"]["nDCG@10"], m["informal"]["nDCG@10"]
        r = paired_bootstrap(i_m, f_m, args.bootstrap)
        gap = i_m.mean() - f_m.mean()
        gaps[name] = {"formal": float(f_m.mean()), "informal": float(i_m.mean()), "gap": float(gap),
                      "gap_pct": float(100 * gap / f_m.mean()), "narrowing_vs_base": float(100 * (gap - base_gap)),
                      "p_informal_vs_formal": r["p"], "ci95": [r["ci95_low"], r["ci95_high"]]}
        g = gaps[name]
        lines.append(f"| {name} | {g['formal']:.4f} | {g['informal']:.4f} | {100 * g['gap']:+.2f} | "
                     f"{g['gap_pct']:+.1f}% | {g['narrowing_vs_base']:+.2f} | {r['p']:.3f}"
                     f"{'*' if r['p'] < 0.05 else ''} |")

    text = "\n".join(lines) + "\n"
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.report_name or ("tuned" if args.tuned else "harmonized")
    with open(os.path.join(args.out_dir, f"report_{tag}_{args.split}.md"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out_dir, f"report_{tag}_{args.split}.json"), "w") as f:
        json.dump({"split": args.split, "n_queries": len(qids), "harmonised": not args.tuned,
                   "results": {n: {s: {k: float(v[k].mean()) for k in order} for s, v in m.items()}
                               for n, m in systems.items()},
                   "gap": gaps, "bootstrap_samples": args.bootstrap}, f, indent=2)
    print("\n" + text)


if __name__ == "__main__":
    main()
