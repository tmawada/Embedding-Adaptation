"""One test-split table for every method trained by run_all.py (the apple-to-apple comparison).

Every row of an encoder is scored against that encoder's frozen 500k-passage index (bge_m3: methods/cache) on the
same test queries, with the same metrics (methods/eval.py: nDCG@10, MRR@10, Recall@100) and paired bootstrap tests
against that encoder's base model. Each method contributes its dev-selected checkpoint (TSDAE: its last step);
rows whose artefacts are missing are skipped.

  python3 comparison/evaluate_all.py                    # -> comparison/outputs/report_test.{md,json}
  python3 comparison/evaluate_all.py --encoder me5      # -> comparison/outputs/report_test_me5.{md,json}
  python3 comparison/evaluate_all.py --encoder compare  # both, then -> comparison/outputs/report_encoders_test.{md,json}

Seed stability (--std true, after run_all.py --std true): every trained row is scored once per --seeds seed and
reported as mean and standard deviation across seeds; the references (base, LLM rewrite) are deterministic and
scored once. Missing seed runs are listed, not fatal.

  python3 comparison/evaluate_all.py --std true                    # -> report_std_test.{md,json}
  python3 comparison/evaluate_all.py --encoder compare --std true  # + report_encoders_std_test.{md,json}

Extra test sets (--testset, after comparison/prepare_testset.py): the same checkpoints, every seed, scored on
style_pilot (the test queries that also have a varied-style informal rewrite) and miracl_dev_formal (official MIRACL-id
dev queries, which the encoders were never trained on; formal only, mpnet base as the yardstick).

  python3 comparison/evaluate_all.py --testset all   # -> report_testsets.{md,json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "methods", "query_distillation"))

from preprocessing.dataset import EmbeddingStore  # noqa: E402
from methods.encoders import ENCODERS, GenericEncoder  # noqa: E402
from methods.eval import METRICS, Evaluator, paired_bootstrap  # noqa: E402
from methods.model import BGEM3Encoder  # noqa: E402
from evaluate import adapt, load_adapter  # noqa: E402  (methods/query_distillation/evaluate.py)
import run_all  # noqa: E402
from run_all import DATASETS, PROTOCOL, SEEDS, artifacts, cache_dir, data_dir  # noqa: E402
from prepare_testset import DEV_DATA, TESTSETS, testset_dir, tower_dir  # noqa: E402

LLM = "gemini-3.5-flash-lite"
TAU_NOTE = ("mE5-large-instruct's cosine scores are compressed (relevant and top-ranked non-relevant passages both "
            "score ~0.89, random ones ~0.76), while the InfoNCE / distillation temperature (tau 0.05) was set for "
            "BGE-M3. The protocol is kept identical on purpose, so mE5 may be at a disadvantage on those methods.")


# Informal style scored as the "informal" column (--informal): data/official has two for its test queries.
INFORMAL = {"style": "template"}
INFORMAL_FILE = lambda: "informal_emb.npy" if INFORMAL["style"] == "template" else "informal_varied_emb.npy"


def report_name(enc: str, std: bool = False) -> str:
    """run_all.report_name (with the dataset tag) + _varied when the varied informal style is scored."""
    return run_all.report_name(enc, std) + ("_varied" if INFORMAL["style"] == "varied" else "")


def dataset_lines() -> list:
    """Header lines naming the dataset layout; none for miracl_train, so those reports stay as they were."""
    if run_all.DATASET == "miracl_train":
        return []
    style = {"template": "template style (the style of every training query, mostly '... sih')",
             "varied": "varied style (sampled per query: slang, abbreviations, typos, word order)"}[INFORMAL["style"]]
    return [f"- Dataset:   {data_dir()}: trained on official MIRACL-id train, checkpoint on our dev split, tested",
            "             on official MIRACL-id dev queries (never used to train the encoders)",
            f"- Informal:  {style}"]


def base_label(enc: str) -> str:
    return f"{ENCODERS[enc].label} base (zero-shot)"


def rows(enc: str, seed: int = PROTOCOL["seed"]) -> list:
    """(label, kind, spec); kind decides how the formal / informal test embeddings are produced."""
    A = artifacts(enc, seed)
    llm_dir = os.path.join("methods", "llm_rewrite", "cache", *([run_all.DATASET] if run_all.dtag() else []))
    return [
        (base_label(enc), "base", {}),
        ("LLM rewrite", "llm", {
            "informal": os.path.join(llm_dir, f"{LLM}_test{'_varied' if INFORMAL['style'] == 'varied' else ''}.jsonl"),
            "formal": os.path.join(llm_dir, f"{LLM}_test_formal.jsonl")}),
        ("Query-DANN (bottleneck)", "adapter", {"ckpt": A["query_dann"]}),
        ("Query-DANN (linear)", "adapter", {"ckpt": A["query_dann_linear"]}),
        ("Query distillation score (bottleneck)", "adapter", {"ckpt": A["query_distillation"]}),
        ("Query distillation score (linear)", "adapter", {"ckpt": A["query_distillation_linear"]}),
        ("TSDAE-SAPT (task queries), final", "tower",
         {"cache": A["tsdae_sapt_queries_final_cache"], "model": A["tsdae_sapt_queries_final"]}),
        ("Query InfoNCE", "tower", {"cache": A["query_infonce_cache"], "model": A["query_infonce_model"]}),
    ]


def notes(enc: str, label: str = None, dim=None) -> dict:
    label, d = label or ENCODERS[enc].label, dim or ENCODERS[enc].dim
    return {
        "LLM rewrite": f"{LLM}, cached rewrites of the test queries (no API calls); not trained. Its formal column "
                       f"rewrites the formal queries too, since a deployed rewriter sees every query.",
        "Query-DANN (bottleneck)": f"the default adapter: zero-initialised residual bottleneck {d} -> 128 -> {d} on "
                                   f"the frozen {label} query vector. Same adapter in \"Query distillation score "
                                   f"(bottleneck)\".",
        "Query-DANN (linear)": f"--adapter linear: a single {d} x {d} linear map initialised to the identity. Same "
                               "adapter in \"Query distillation score (linear)\".",
        "Query InfoNCE": "--objective infonce: the full encoder fine-tuned with InfoNCE on informal queries against "
                         "the frozen passage index, without the TSDAE reconstruction term that the method adds by "
                         "default (weight 0.1).",
        "TSDAE-SAPT (task queries), final": "the encoder after the last training step, NOT the best dev checkpoint "
                                            "like the other rows: dev selection keeps step 50, which has barely "
                                            "trained and scores like base. This row shows what TSDAE training itself "
                                            "does to retrieval.",
    }


def r(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def selected_step(*config_paths: str):
    """Step of the dev-selected checkpoint, from whichever config file the trainer wrote."""
    for p in config_paths:
        if os.path.exists(r(p)):
            with open(r(p)) as f:
                step = json.load(f).get("step")
            if step is not None:
                return step
    return None


def read_rewrites(path: str, qids):
    with open(r(path)) as f:
        rows = {x["query_id"]: x["normalized"] for x in map(json.loads, filter(str.strip, f))}
    missing = [q for q in qids if q not in rows]
    return None if missing else [rows[q] for q in qids]


def tower_embeddings(cache: str, ev: Evaluator, store: EmbeddingStore):
    """Formal / informal test vectors from a re-encoded query cache (passages = the encoder's frozen index)."""
    with open(os.path.join(r(cache), "query_ids.json")) as f:
        assert json.load(f) == store.query_ids, f"query order in {cache} differs from the base cache"
    rows = ev.rows.cpu()
    load = lambda name: torch.from_numpy(np.load(os.path.join(r(cache), name)))[rows].to(store.device).float()
    return load("formal_emb.npy"), load(INFORMAL_FILE())


def embed_row(enc: str, kind: str, rspec: dict, ev: Evaluator, store: EmbeddingStore, models: dict):
    """(formal vectors, informal vectors, selected step) of one row, or the reason it is skipped as a string.
    `models` keeps the query encoder that the LLM row loads, so it is loaded once per encoder."""
    spec = ENCODERS[enc]
    if kind == "base":
        return ev.formal(), ev.informal(), None
    if kind == "llm":
        texts = {side: read_rewrites(rspec[side], ev.qids) if os.path.exists(r(rspec[side])) else None
                 for side in ("formal", "informal")}
        if texts["informal"] is None:
            return f"no complete rewrites at {rspec['informal']}"
        if "encoder" not in models:
            models["encoder"] = BGEM3Encoder(spec.model_id) if enc == "bge_m3" else GenericEncoder(spec)
        encoder = models["encoder"]
        if enc == "bge_m3":  # exactly as methods/prepare.py encoded the base queries
            emb = lambda t: encoder.encode(t, max_length=spec.query_max_length, progress=False)
        else:  # the encoder's own pooling and query prefix
            emb = lambda t: encoder.encode_queries(t, progress=False)
        to_t = lambda t: torch.from_numpy(emb(t)).to(store.device).float()
        return to_t(texts["formal"]) if texts["formal"] else ev.formal(), to_t(texts["informal"]), None
    if kind == "adapter":
        if not os.path.exists(os.path.join(r(rspec["ckpt"]), "adapter.pt")):
            return f"no checkpoint at {rspec['ckpt']}"
        adapter = load_adapter(r(rspec["ckpt"]), store.device)
        return (adapt(adapter, ev.formal()), adapt(adapter, ev.informal()),
                selected_step(os.path.join(rspec["ckpt"], "config.json")))
    if kind == "tower":
        if not os.path.exists(os.path.join(r(rspec["cache"]), INFORMAL_FILE())):
            return f"no query cache at {rspec['cache']}"
        z_form, z_inf = tower_embeddings(rspec["cache"], ev, store)
        return z_form, z_inf, selected_step(*(os.path.join(rspec["model"], n)
                                              for n in ("tsdae_info.json", "sapt_config.json", "infonce_config.json")))
    raise ValueError(kind)


def open_store(enc: str) -> EmbeddingStore:
    """The dataset's cache of this encoder; its informal column is the --informal style."""
    store = EmbeddingStore(cache_dir(enc))
    if INFORMAL["style"] == "varied":
        store.informal = store._load(os.path.join(cache_dir(enc), INFORMAL_FILE()))
    return store


def evaluate(enc: str, args) -> dict:
    """Score every row for one encoder, write report_test[_<enc>].{md,json} and return the JSON payload."""
    spec, BASE = ENCODERS[enc], base_label(enc)
    store = open_store(enc)
    with open(os.path.join(cache_dir(enc), "splits.json")) as f:
        ev = Evaluator(store, json.load(f)[args.split], os.path.join(ROOT, data_dir()))
    print(f"[{enc}] {args.split}: {len(ev.qids)} queries, {len(store.doc_ids):,} passages")

    per_query, meta, skipped = {}, {}, {}
    models, row_notes = {}, notes(enc)
    for label, kind, rspec in rows(enc):
        got = embed_row(enc, kind, rspec, ev, store, models)
        if isinstance(got, str):
            skipped[label] = got
            continue
        z_form, z_inf, step = got
        per_query[label] = {"formal": ev.run_per_query(z_form), "informal": ev.run_per_query(z_inf)}
        meta[label] = {"selected_step": step, "supplementary": rspec.get("supplementary", False),
                       "note": row_notes.get(label, "")}
        print(f"  scored {label}")

    base = per_query[BASE]
    b_form, b_inf = base["formal"]["nDCG@10"].mean(), base["informal"]["nDCG@10"].mean()
    results = {}
    for label, pq in per_query.items():
        row = {side: {k: float(pq[side][k].mean()) for k in METRICS} for side in ("formal", "informal")}
        if label != BASE:
            for side in ("formal", "informal"):
                row[f"{side}_vs_base"] = {k: paired_bootstrap(pq[side][k], base[side][k], args.bootstrap) for k in METRICS}
        else:  # the robustness gap itself: informal vs formal queries on the base model
            row["informal_vs_formal"] = {k: paired_bootstrap(pq["informal"][k], pq["formal"][k], args.bootstrap)
                                         for k in METRICS}
        row["gap_closed_pct"] = 100 * (row["informal"]["nDCG@10"] - b_inf) / (b_form - b_inf)
        results[label] = {**row, **meta[label]}

    payload = {"encoder": enc, "encoder_label": spec.label, "model_id": spec.model_id, "base_label": BASE,
               "split": args.split, "n_queries": len(ev.qids), "n_passages": len(store.doc_ids),
               "protocol": PROTOCOL, "bootstrap_samples": args.bootstrap, "results": results, "skipped": skipped}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, report_name(enc).replace("test", args.split))
    with open(out + ".json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(out + ".md", "w") as f:
        f.write(markdown(payload, args))
    print(open(out + ".md").read())
    print(f"-> {os.path.relpath(out, ROOT)}.md / .json")
    del store, ev, models
    torch.cuda.empty_cache()
    return payload


def spread(values: list) -> dict:
    """Mean, sample standard deviation (n - 1; None for a single value), min and max."""
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "min": float(min(values)), "max": float(max(values))}


def aggregate(runs: list) -> dict:
    """Across-seed summary of one row's per-seed results (the dicts written by evaluate_std)."""
    out = {"n": len(runs), "gap_closed_pct": spread([x["gap_closed_pct"] for x in runs])}
    for side in ("formal", "informal"):
        out[side] = {k: spread([x[side][k] for x in runs]) for k in METRICS}
        tests = [x[f"{side}_vs_base"] for x in runs if f"{side}_vs_base" in x]
        out[f"{side}_vs_base_pts"] = spread([100 * b["diff"] for b in tests]) if tests else None
        out[f"{side}_significant"] = {"up": sum(b["p"] < 0.05 and b["diff"] > 0 for b in tests),
                                      "down": sum(b["p"] < 0.05 and b["diff"] < 0 for b in tests)}
    return out


def evaluate_std(enc: str, args) -> dict:
    """Seed stability for one encoder: every trained row once per seed (the references once), mean / std across seeds;
    writes report_std_test[_<enc>].{md,json} and returns the JSON payload."""
    spec, BASE = ENCODERS[enc], base_label(enc)
    store = open_store(enc)
    with open(os.path.join(cache_dir(enc), "splits.json")) as f:
        ev = Evaluator(store, json.load(f)[args.split], os.path.join(ROOT, data_dir()))
    print(f"[{enc}] {args.split}: {len(ev.qids)} queries, {len(store.doc_ids):,} passages, seeds {args.seeds}")
    models, row_notes = {}, notes(enc)
    per_query = lambda z_form, z_inf: {"formal": ev.run_per_query(z_form), "informal": ev.run_per_query(z_inf)}

    refs, skipped = {}, {}
    for label, kind, rspec in rows(enc):
        if kind in ("base", "llm"):  # deterministic: no training, no seed
            got = embed_row(enc, kind, rspec, ev, store, models)
            if isinstance(got, str):
                skipped[label] = got
                continue
            refs[label] = per_query(*got[:2])
    base = refs[BASE]
    b_form, b_inf = base["formal"]["nDCG@10"].mean(), base["informal"]["nDCG@10"].mean()

    def summary(pq: dict, step) -> dict:
        row = {side: {k: float(pq[side][k].mean()) for k in METRICS} for side in ("formal", "informal")}
        if pq is not base:  # nDCG@10 test against base, per seed
            for side in ("formal", "informal"):
                row[f"{side}_vs_base"] = paired_bootstrap(pq[side]["nDCG@10"], base[side]["nDCG@10"], args.bootstrap)
        row["gap_closed_pct"] = 100 * (row["informal"]["nDCG@10"] - b_inf) / (b_form - b_inf)
        row["selected_step"] = step
        return row

    results = {label: {"reference": True, "seeds": {}, "note": row_notes.get(label, ""),
                       "aggregate": aggregate([summary(pq, None)])} for label, pq in refs.items()}
    for seed in args.seeds:
        for label, kind, rspec in rows(enc, seed):
            if kind in ("base", "llm"):
                continue
            got = embed_row(enc, kind, rspec, ev, store, models)
            if isinstance(got, str):
                skipped.setdefault(label, {})[str(seed)] = got
                continue
            row = results.setdefault(label, {"reference": False, "seeds": {}, "note": row_notes.get(label, "")})
            row["seeds"][str(seed)] = summary(per_query(*got[:2]), got[2])
            print(f"  scored {label}, seed {seed}")
    for label, row in results.items():
        if not row["reference"]:
            row["aggregate"] = aggregate(list(row["seeds"].values()))

    payload = {"encoder": enc, "encoder_label": spec.label, "model_id": spec.model_id, "base_label": BASE,
               "split": args.split, "n_queries": len(ev.qids), "n_passages": len(store.doc_ids),
               "protocol": PROTOCOL, "seeds": args.seeds, "bootstrap_samples": args.bootstrap, "results": results,
               "skipped": skipped}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, report_name(enc, std=True).replace("test", args.split))
    with open(out + ".json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(out + ".md", "w") as f:
        f.write(std_markdown(payload, args))
    print(open(out + ".md").read())
    print(f"-> {os.path.relpath(out, ROOT)}.md / .json")
    del store, ev, models
    torch.cuda.empty_cache()
    return payload


# --------------------------------------------------------------------------- #
# Extra test sets (--testset): caches from comparison/prepare_testset.py
# --------------------------------------------------------------------------- #
SIDES = {"style_pilot": ("formal", "template", "varied"), "miracl_dev_formal": ("formal",)}
SIDE_FILES = {"formal": "formal_emb.npy", "template": "informal_emb.npy", "varied": "informal_varied_emb.npy"}
YARDSTICK = "mpnet"  # never trained on MIRACL: how much harder the unseen queries are, without any memory


def sig_counts(tests: list) -> dict:
    return {"up": sum(b["p"] < 0.05 and b["diff"] > 0 for b in tests),
            "down": sum(b["p"] < 0.05 and b["diff"] < 0 for b in tests)}


def side_vectors(cache: str, name: str, ev: Evaluator, store: EmbeddingStore) -> torch.Tensor:
    """One query file of a test-set cache, in the evaluator's query order."""
    with open(os.path.join(cache, "query_ids.json")) as f:
        assert json.load(f) == store.query_ids, f"query order in {cache} differs from the test-set cache"
    return torch.from_numpy(np.load(os.path.join(cache, name)))[ev.rows.cpu()].to(store.device).float()


def testset_embed(testset: str, enc: str, kind: str, rspec: dict, ev: Evaluator, store: EmbeddingStore):
    """{side: query vectors} of one row on a test set, or the reason it is skipped as a string."""
    sides, base_dir = SIDES[testset], testset_dir(testset, enc)
    if kind == "base":
        return {s: side_vectors(base_dir, SIDE_FILES[s], ev, store) for s in sides}
    if kind == "llm":
        return "no LLM rewrites of these queries (they would need API calls)"
    if kind == "adapter":
        if not os.path.exists(os.path.join(r(rspec["ckpt"]), "adapter.pt")):
            return f"no checkpoint at {rspec['ckpt']}"
        adapter = load_adapter(r(rspec["ckpt"]), store.device)
        return {s: adapt(adapter, side_vectors(base_dir, SIDE_FILES[s], ev, store)) for s in sides}
    if kind == "tower":
        cache = tower_dir(testset, enc, rspec["cache"])
        if not os.path.exists(os.path.join(cache, SIDE_FILES[sides[-1]])):
            return f"its fine-tuned query encoders are gone ({os.path.dirname(os.path.dirname(rspec['model']))}/)"
        return {s: side_vectors(cache, SIDE_FILES[s], ev, store) for s in sides}
    raise ValueError(kind)


def evaluate_style_pilot(enc: str, args) -> dict:
    """Same questions, two informal styles: does a model's gain over base hold under the varied style?"""
    store = EmbeddingStore(testset_dir("style_pilot", enc))
    ev = Evaluator(store, store.query_ids, os.path.join(ROOT, "data"))
    print(f"[{enc}] style_pilot: {len(ev.qids)} queries, seeds {args.seeds}")
    sides, nd = SIDES["style_pilot"], lambda pq, s: pq[s]["nDCG@10"]
    per_query = lambda vectors: {s: ev.run_per_query(z) for s, z in vectors.items()}
    boot = lambda a, b: paired_bootstrap(a, b, args.bootstrap)

    base = per_query(testset_embed("style_pilot", enc, "base", {}, ev, store))
    gap = {s: nd(base, "formal").mean() - nd(base, s).mean() for s in ("template", "varied")}
    out = {"n_queries": len(ev.qids),
           "base": {"scores": {s: {k: float(base[s][k].mean()) for k in METRICS} for s in sides},
                    "template_vs_formal": boot(nd(base, "template"), nd(base, "formal")),
                    "varied_vs_formal": boot(nd(base, "varied"), nd(base, "formal")),
                    "varied_vs_template": boot(nd(base, "varied"), nd(base, "template"))},
           "models": {}, "skipped": {}}

    runs = {}  # label -> {seed: per-query metrics per side}
    for seed in args.seeds:
        for label, kind, rspec in rows(enc, seed):
            if kind == "base" or (kind == "llm" and seed != args.seeds[0]):
                continue
            got = testset_embed("style_pilot", enc, kind, rspec, ev, store)
            if isinstance(got, str):
                if kind == "llm":
                    out["skipped"][label] = got
                else:
                    out["skipped"].setdefault(label, {})[str(seed)] = got
                continue
            runs.setdefault(label, {})[seed] = per_query(got)

    # Varied gain minus template gain, per query: does the gain shrink when the informal style changes?
    shift = lambda pq: (nd(pq, "varied") - nd(base, "varied"), nd(pq, "template") - nd(base, "template"))
    for label, by_seed in runs.items():
        seeds = {}
        for seed, pq in by_seed.items():
            seeds[str(seed)] = {"scores": {s: {k: float(pq[s][k].mean()) for k in METRICS} for s in sides},
                                **{f"{s}_vs_base": boot(nd(pq, s), nd(base, s)) for s in sides},
                                "varied_minus_template": boot(*shift(pq)),
                                "gap_closed_pct": {s: 100 * (nd(pq, s).mean() - nd(base, s).mean()) / gap[s]
                                                   for s in ("template", "varied")}}
        # The seed-averaged model (per-query scores averaged over seeds): one CI for the query sampling.
        avg = {s: {"nDCG@10": np.mean([nd(pq, s) for pq in by_seed.values()], axis=0)} for s in sides}
        tests = lambda key: [x[key] for x in seeds.values()]
        out["models"][label] = {"seeds": seeds, "aggregate": {
            "n": len(seeds),
            **{f"{s}_vs_base_pts": spread([100 * b["diff"] for b in tests(f"{s}_vs_base")]) for s in sides},
            **{f"{s}_significant": sig_counts(tests(f"{s}_vs_base")) for s in sides},
            "varied_minus_template_pts": spread([100 * b["diff"] for b in tests("varied_minus_template")]),
            "varied_minus_template_significant": sig_counts(tests("varied_minus_template")),
            "gap_closed_pct": {s: spread([x["gap_closed_pct"][s] for x in seeds.values()]) for s in ("template", "varied")},
            "seed_averaged": {**{f"{s}_vs_base": boot(nd(avg, s), nd(base, s)) for s in sides},
                              "varied_minus_template": boot(*shift(avg))}}}
        print(f"  scored {label}, {len(seeds)} seed(s)")
    del store, ev
    torch.cuda.empty_cache()
    return out


def drop_bootstrap(seen: dict, unseen: dict, n_samples: int, seed: int = 0) -> dict:
    """Drop from seen to unseen queries, mean(seen) - mean(unseen), per encoder: absolute (nDCG@10) and relative to the
    seen score. The two query sets differ, so each is resampled on its own (unpaired); every encoder gets the same draw,
    which keeps the encoders paired inside each set, so their difference ("memory effect", first encoder minus each
    other one) gets its own CI. p as in methods/eval.py paired_bootstrap: null-shifted, (count + 1) / (n + 1)."""
    names = list(seen)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(seen[names[0]]), size=(n_samples, len(seen[names[0]])))
    j = rng.integers(0, len(unseen[names[0]]), size=(n_samples, len(unseen[names[0]])))

    def summary(obs: float, boot: np.ndarray) -> dict:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = (np.sum(np.abs(boot - obs) >= abs(obs)) + 1) / (n_samples + 1)
        return {"diff": float(obs), "ci95_low": float(lo), "ci95_high": float(hi), "p": float(p)}

    obs, boot = {}, {}
    for e in names:
        s, u = seen[e].mean(), unseen[e].mean()
        bs, bu = seen[e][i].mean(1), unseen[e][j].mean(1)
        obs[e] = {"abs": s - u, "rel": (s - u) / s}
        boot[e] = {"abs": bs - bu, "rel": (bs - bu) / bs}
    out = {e: {k: summary(obs[e][k], boot[e][k]) for k in ("abs", "rel")} for e in names}
    for e in names[1:]:
        out[f"{names[0]}_minus_{e}"] = {k: summary(obs[names[0]][k] - obs[e][k], boot[names[0]][k] - boot[e][k])
                                        for k in ("abs", "rel")}
    return out


def evaluate_dev_formal(enc: str, args) -> dict:
    """Formal queries the encoder was trained on (our test split) vs never saw (official MIRACL-id dev)."""
    def open_sets(e: str):
        """(seen evaluator + store, unseen evaluator + store); the unseen store shares the seen store's index."""
        spec = ENCODERS[e]
        seen_store = EmbeddingStore(spec.cache_dir)
        with open(os.path.join(spec.cache_dir, "splits.json")) as f:
            seen_ev = Evaluator(seen_store, json.load(f)["test"], os.path.join(ROOT, "data"))
        dev_store = EmbeddingStore(testset_dir("miracl_dev_formal", e), load_corpus=False)
        dev_store.corpus = seen_store.corpus
        return (seen_ev, seen_store), (Evaluator(dev_store, dev_store.query_ids, DEV_DATA), dev_store)

    encoders = [enc] + [e for e in (YARDSTICK,) if e != enc and
                        os.path.exists(os.path.join(testset_dir("miracl_dev_formal", e), "formal_emb.npy"))]
    base, sizes = {}, {}
    for e in encoders:
        (seen_ev, _), (dev_ev, _) = open_sets(e)
        base[e] = {"seen": seen_ev.run_per_query(seen_ev.formal()), "unseen": dev_ev.run_per_query(dev_ev.formal())}
        sizes = {"n_seen": len(seen_ev.qids), "n_unseen": len(dev_ev.qids)}
        del seen_ev, dev_ev
        torch.cuda.empty_cache()
    print(f"[{enc}] miracl_dev_formal: {sizes['n_seen']} seen vs {sizes['n_unseen']} unseen formal queries; "
          f"encoders {', '.join(encoders)}")
    out = {**sizes, "encoders": encoders,
           "base": {e: {w: {k: float(base[e][w][k].mean()) for k in METRICS} for w in ("seen", "unseen")}
                    for e in encoders},
           "drop": drop_bootstrap({e: base[e]["seen"]["nDCG@10"] for e in encoders},
                                  {e: base[e]["unseen"]["nDCG@10"] for e in encoders}, args.bootstrap),
           "models": {}, "skipped": {}}

    # Formal preservation of each trained model, on seen and on unseen formal queries.
    (seen_ev, seen_store), (dev_ev, dev_store) = open_sets(enc)
    models = {}
    for seed in args.seeds:
        for label, kind, rspec in rows(enc, seed):
            if kind == "base" or (kind == "llm" and seed != args.seeds[0]):
                continue
            seen = embed_row(enc, kind, rspec, seen_ev, seen_store, models) if kind != "llm" else None
            unseen = testset_embed("miracl_dev_formal", enc, kind, rspec, dev_ev, dev_store)
            why = next((x for x in (unseen, seen) if isinstance(x, str)), None)
            if kind == "llm" or why:
                if kind == "llm":
                    out["skipped"][label] = unseen
                else:
                    out["skipped"].setdefault(label, {})[str(seed)] = why
                continue
            run = {"seen": seen_ev.run_per_query(seen[0])["nDCG@10"],
                   "unseen": dev_ev.run_per_query(unseen["formal"])["nDCG@10"]}
            row = out["models"].setdefault(label, {"seeds": {}})
            row["seeds"][str(seed)] = {w: {"score": float(run[w].mean()),
                                           "vs_base": paired_bootstrap(run[w], base[enc][w]["nDCG@10"], args.bootstrap)}
                                       for w in ("seen", "unseen")}
    for label, row in out["models"].items():
        tests = lambda w: [x[w]["vs_base"] for x in row["seeds"].values()]
        row["aggregate"] = {"n": len(row["seeds"]),
                            **{f"{w}_vs_base_pts": spread([100 * b["diff"] for b in tests(w)]) for w in ("seen", "unseen")},
                            **{f"{w}_significant": sig_counts(tests(w)) for w in ("seen", "unseen")}}
        print(f"  scored {label}, {row['aggregate']['n']} seed(s)")
    del seen_ev, dev_ev, seen_store, dev_store, models
    torch.cuda.empty_cache()
    return out


def evaluate_testsets(enc: str, args) -> dict:
    """Score one encoder's checkpoints on the extra test sets; writes report_testsets[_<enc>].{md,json}
    (report_<testset>[_<enc>] when only one test set is asked for)."""
    names = TESTSETS if args.testset == "all" else (args.testset,)
    spec = ENCODERS[enc]
    payload = {"encoder": enc, "encoder_label": spec.label, "model_id": spec.model_id, "base_label": base_label(enc),
               "seeds": args.seeds, "bootstrap_samples": args.bootstrap}
    for ts, run in (("style_pilot", evaluate_style_pilot), ("miracl_dev_formal", evaluate_dev_formal)):
        if ts not in names:
            continue
        if not os.path.exists(os.path.join(testset_dir(ts, enc), "formal_emb.npy")):
            print(f"[{enc}] {ts}: no test-set cache, run comparison/prepare_testset.py --encoder {enc}")
            payload[ts] = None
            continue
        payload[ts] = run(enc, args)
    os.makedirs(args.out_dir, exist_ok=True)
    name = "report_testsets" if args.testset == "all" else f"report_{args.testset}"
    out = os.path.join(args.out_dir, name if enc == "bge_m3" else f"{name}_{enc}")
    with open(out + ".json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(out + ".md", "w") as f:
        f.write(testset_markdown(payload, args))
    print(open(out + ".md").read())
    print(f"-> {os.path.relpath(out, ROOT)}.md / .json")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="bge_m3", choices=[*ENCODERS, "compare"],
                    help="One encoder's report, or compare = every encoder's report + the side-by-side one")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--std", default="false", choices=["true", "false"],
                    help="Seed-stability report: every trained row once per --seeds seed, mean / std across seeds")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS, metavar="SEED",
                    help=f"Seeds for --std true / --testset, any of {' '.join(map(str, SEEDS))} (default: all five)")
    ap.add_argument("--testset", default=None, choices=[*TESTSETS, "all"],
                    help="Score the extra test sets of comparison/prepare_testset.py (every seed) instead of --split")
    ap.add_argument("--dataset", default="miracl_train", choices=DATASETS,
                    help="Dataset layout (comparison/run_all.py): miracl_train (our split) or official (data/official)")
    ap.add_argument("--informal", default=None, choices=["template", "varied", "both"],
                    help="Informal style scored (data/official has two); default: template, both for --dataset official")
    args = ap.parse_args()
    std = args.std == "true"
    args.seeds = sorted(set(args.seeds))
    run_all.set_dataset(args.dataset)
    styles = {"both": ["template", "varied"], None: ["template", "varied"] if args.dataset == "official"
              else ["template"]}.get(args.informal, [args.informal])
    if args.testset:  # the extra test sets score every style they have in one report
        styles = ["template"]
    if len(styles) > 1:  # one report per style
        for style in styles:
            INFORMAL["style"] = style
            main_one(args, std)
        return
    INFORMAL["style"] = styles[0]
    main_one(args, std)


def main_one(args, std: bool):
    if args.testset:
        for enc in (ENCODERS if args.encoder == "compare" else [args.encoder]):
            evaluate_testsets(enc, args)
        return
    run = evaluate_std if std else evaluate

    if args.encoder != "compare":
        run(args.encoder, args)
        return
    payloads = [run(enc, args) for enc in ENCODERS]
    out = os.path.join(args.out_dir, f"report_encoders_{'std_' if std else ''}{args.split}")
    with open(out + ".json", "w") as f:
        json.dump({"split": args.split, "protocol": PROTOCOL, **({"seeds": args.seeds} if std else {}),
                   "bootstrap_samples": args.bootstrap, "encoders": {p["encoder"]: p for p in payloads}}, f, indent=2)
    with open(out + ".md", "w") as f:
        f.write((std_compare_markdown if std else compare_markdown)(payloads, args))
    print(open(out + ".md").read())
    print(f"-> {os.path.relpath(out, ROOT)}.md / .json")


def table(headers, rows) -> list:
    """Markdown table padded so the columns also line up in a plain-text editor; columns after the first are
    right-aligned numbers."""
    widths = [max(len(r[i]) for r in [headers, *rows]) for i in range(len(headers))]
    line = lambda r: "| " + " | ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                      for i, (c, w) in enumerate(zip(r, widths))) + " |"
    rule = "|" + "|".join("-" * (w + 2) if i == 0 else "-" * (w + 1) + ":" for i, w in enumerate(widths)) + "|"
    return [line(headers), rule, *map(line, rows)]


wrap = lambda text, indent="": textwrap.fill(text, 100, subsequent_indent=indent, break_on_hyphens=False)
# A trailing space where there is no star keeps the decimal points aligned.
star = lambda b: "*" if b["p"] < 0.05 else " "
pts = lambda b: "- " if b is None else f"{100 * b['diff']:+.2f}{star(b)}"


def protocol_lines(seeds=None) -> list:
    p = PROTOCOL
    seed = (f"- Seeds:       {', '.join(map(str, seeds))} (one training run per seed and model)" if seeds
            else f"- Seed:        {p['seed']}")
    return ["## Training protocol (every trained model)", "",
            f"- Steps:       {p['total_steps']} optimiser steps, effective batch {p['effective_batch']}",
            f"- Optimiser:   {p['optimizer']}, lr {p['lr']:g}, weight decay {p['weight_decay']}, "
            f"grad clip {p['grad_clip']}",
            f"- Schedule:    cosine down to {p['lr_min']:g} after {int(100 * p['warmup_ratio'])}% linear warmup",
            f"- Precision:   {p['precision']}, max_length {p['max_length']} (encoder fine-tuning)",
            seed,
            f"- Checkpoint:  {p['selection']}, checked every {p['eval_every']} steps"]


def markdown(payload: dict, args) -> str:
    """Plain ASCII, lines within ~100 characters: the report has to read well in Notepad, not only rendered."""
    results, skipped, BASE = payload["results"], payload["skipped"], payload["base_label"]
    label, n_queries, n_passages = payload["encoder_label"], payload["n_queries"], payload["n_passages"]
    # Base first as the reference, then the best informal nDCG@10 first; every table uses this order.
    order = [BASE] + sorted((l for l in results if l != BASE), key=lambda l: -results[l]["informal"]["nDCG@10"])
    vs_base = lambda l, side: results[l].get(f"{side}_vs_base", {}).get("nDCG@10")

    lines = [f"# Harmonized comparison: MIRACL-id {args.split} set", "",
             f"- Encoder:   {label} ({payload['model_id']})",
             f"- Queries:   {n_queries} {args.split} queries, each in a formal and an informal version",
             *dataset_lines(),
             f"- Passages:  {n_passages:,}, one frozen {label} index searched by every row",
             "- Scores:    nDCG@10 unless stated otherwise; pts = nDCG@10 x 100", "",
             *protocol_lines(), "",
             "## 1. Summary", "",
             wrap("Sorted by informal nDCG@10. Gain and Change = difference from base in pts (* = p < 0.05, "
                  f"paired bootstrap, {args.bootstrap} samples). Gap closed = share of the base model's "
                  "formal-informal gap recovered on informal queries."), "",
             *table(["Model", "Informal", "Gain (pts)", "Formal", "Change (pts)", "Gap closed"],
                    [[l, f"{results[l]['informal']['nDCG@10']:.4f}", pts(vs_base(l, "informal")),
                      f"{results[l]['formal']['nDCG@10']:.4f}", pts(vs_base(l, "formal")),
                      "-" if l == BASE else f"{results[l]['gap_closed_pct']:.1f}%"] for l in order])]
    for n, side in ((2, "informal"), (3, "formal")):
        lines += ["", f"## {n}. All metrics, {side} queries", "",
                  *table(["Model", *METRICS], [[l, *(f"{results[l][side][k]:.4f}" for k in METRICS)] for l in order])]
    rows = []
    for l in order[1:]:
        bi, bf, step = vs_base(l, "informal"), vs_base(l, "formal"), results[l]["selected_step"]
        rows.append([l, pts(bi), f"[{100 * bi['ci95_low']:+.2f}, {100 * bi['ci95_high']:+.2f}]",
                     f"{bi['p']:.3f}{star(bi)}", pts(bf), f"{bf['p']:.3f}{star(bf)}", "-" if step is None else str(step)])
    lines += ["", "## 4. Significance vs base and checkpoint step", "",
              wrap(f"Informal / Formal = nDCG@10 difference from base in pts, paired bootstrap with {args.bootstrap} "
                   "samples; "
                   "* = p < 0.05. Step = optimiser step of the scored model: the best dev checkpoint unless the "
                   "notes say otherwise (- = not trained)."), "",
              *table(["Model", "Informal", "95% CI", "p", "Formal", "p", "Step"], rows)]
    row_notes = [wrap(f"- {l}: {results[l]['note']}", "  ") for l in order if results[l]["note"]]
    if payload["encoder"] == "me5":
        row_notes.append(wrap(f"- Temperature: {TAU_NOTE}", "  "))
    if row_notes:
        lines += ["", "## Notes", "", *row_notes]
    if skipped:
        lines += ["", "## Skipped", "", *[wrap(f"- {l}: {why}", "  ") for l, why in skipped.items()]]
    return "\n".join(lines) + "\n"


SHORT = {"bge_m3": "BGE-M3", "me5": "mE5", "mpnet": "mpnet", "arctic": "Arctic"}
DIM_NOTE = "d = the encoder's output size (" + ", ".join(f"{s.label} {s.dim}" for s in ENCODERS.values()) + ")"


def by_model(methods: list, payloads: list, names: list, cells) -> list:
    """One row per model and encoder, the model name only on its first row: scales to any number of encoders."""
    return [[l if i == 0 else "", n, *cells(p, l)] for l in methods for i, (n, p) in enumerate(zip(names, payloads))]


def compare_markdown(payloads: list, args) -> str:
    """Every method's effect on each encoder, measured against that encoder's own base."""
    names = [SHORT.get(p["encoder"], p["encoder"]) for p in payloads]
    methods = [l for l, kind, _ in rows(payloads[0]["encoder"]) if kind != "base"]
    res = lambda p, l: p["results"].get(l)
    vs_base = lambda p, l, side: (res(p, l) or {}).get(f"{side}_vs_base", {}).get("nDCG@10")
    score = lambda p, l, side: f"{res(p, l)[side]['nDCG@10']:.4f}" if res(p, l) else "n/a"
    gap = lambda p, l: f"{res(p, l)['gap_closed_pct']:.1f}%" if res(p, l) else "n/a"

    lines = [f"# Encoder comparison: MIRACL-id {args.split} set", "",
             *[f"- {n + ':':<10} {p['encoder_label']} ({p['model_id']})" for n, p in zip(names, payloads)],
             f"- Queries:   {payloads[0]['n_queries']} {args.split} queries, each in a formal and an informal version",
             "- Passages:  each encoder searches its own frozen index of the same "
             f"{payloads[0]['n_passages']:,} passages",
             "- Scores:    nDCG@10; pts = nDCG@10 x 100; every gain is measured against THAT encoder's own base", "",
             *protocol_lines(), "",
             "## 1. Zero-shot robustness of the base encoders", "",
             wrap("Drop = informal minus formal nDCG@10 on the base model, in pts (* = p < 0.05, paired "
                  f"bootstrap, {args.bootstrap} samples): the gap the methods try to close."), "",
             *table(["Encoder", "Formal", "Informal", "Drop (pts)"],
                    [[p["encoder_label"], score(p, p["base_label"], "formal"), score(p, p["base_label"], "informal"),
                      pts(p["results"][p["base_label"]]["informal_vs_formal"]["nDCG@10"])] for p in payloads]),
             "", "## 2. Informal and formal queries per model and encoder", "",
             wrap("nDCG@10; Gain and Change = difference from that encoder's base in pts (* = p < 0.05). Gap closed = "
                  "share of that encoder's formal-informal gap recovered on informal queries."), "",
             *table(["Model", "Encoder", "Informal", "Gain", "Gap closed", "Formal", "Change"],
                    by_model(methods, payloads, names, lambda p, l: (
                        score(p, l, "informal"), pts(vs_base(p, l, "informal")), gap(p, l), score(p, l, "formal"),
                        pts(vs_base(p, l, "formal"))))),
             "", "## Notes", "",
             wrap("- Details per encoder (MRR@10, Recall@100, 95% CIs, p-values, selected steps): "
                  + ", ".join(report_name(p["encoder"]).replace("test", args.split) + ".md" for p in payloads)
                  + ".", "  "),
             *[wrap(f"- {l}: {note}", "  ") for l, note in notes(payloads[0]["encoder"], "base", "d").items()],
             wrap(f"- Adapter size: {DIM_NOTE}.", "  "),
             wrap(f"- Temperature: {TAU_NOTE}", "  ")]
    return "\n".join(lines) + "\n"


STD_READING = ("Std = sample standard deviation across seeds (n - 1), in pts; it is the same for the score and for "
               "the gain, because the base row does not depend on the seed. Sig. up/down = how many seeds are "
               "significantly better / worse than base on their own (paired bootstrap on nDCG@10, p < 0.05). A model "
               "is stable when its std is small next to its gain and all seeds agree in Sig. up/down.")


def seed_note(reports: str) -> str:
    return (f"Only the training seed changes between runs (initialisation, batch order, sampled hard negatives, "
            f"TSDAE noise, dropout); queries, split, caches and protocol are identical. Seed {PROTOCOL['seed']} is "
            f"the run in {reports}. Base and LLM rewrite involve no training: scored once, so they have no std.")


def missing_lines(skipped: dict) -> list:
    why = lambda w: w if isinstance(w, str) else "; ".join(f"seed {s}: {x}" for s, x in w.items())
    return [wrap(f"- {l}: {why(w)}", "  ") for l, w in skipped.items()]


def std_markdown(payload: dict, args) -> str:
    """Seed-stability report for one encoder; plain ASCII like markdown()."""
    results, skipped, BASE, seeds = payload["results"], payload["skipped"], payload["base_label"], payload["seeds"]
    label = payload["encoder_label"]
    agg = lambda l: results[l]["aggregate"]
    order = [BASE] + sorted((l for l in results if l != BASE), key=lambda l: -agg(l)["informal"]["nDCG@10"]["mean"])
    trained = [l for l in order if not results[l]["reference"]]

    def diff(l, side, key):
        """Mean or std of the nDCG@10 difference from base, in pts."""
        d = agg(l)[f"{side}_vs_base_pts"]
        if d is None or d[key] is None:
            return "-"
        return f"{d[key]:+.2f}" if key == "mean" else f"{d[key]:.2f}"

    count = lambda l: "-" if results[l]["reference"] else str(agg(l)["n"])
    sig = lambda l, side: "-" if results[l]["reference"] else "{up}/{down}".format(**agg(l)[f"{side}_significant"])
    gap = lambda l: "-" if l == BASE else f"{agg(l)['gap_closed_pct']['mean']:.1f}%"
    mean_std = lambda s: f"{s['mean']:.4f}" if s["std"] is None else f"{s['mean']:.4f} ({s['std']:.4f})"
    run = lambda l, s: results[l]["seeds"].get(str(s))

    def per_seed(side):
        out = []
        for l in trained:
            got = [run(l, s)[side]["nDCG@10"] for s in seeds if run(l, s)]
            out.append([l, *(f"{run(l, s)[side]['nDCG@10']:.4f}" if run(l, s) else "-" for s in seeds),
                        f"{100 * (max(got) - min(got)):.2f}" if len(got) > 1 else "-"])
        return table(["Model", *map(str, seeds), "Range (pts)"], out)

    lines = [f"# Seed stability: MIRACL-id {args.split} set", "",
             f"- Encoder:   {label} ({payload['model_id']})",
             f"- Queries:   {payload['n_queries']} {args.split} queries, each in a formal and an informal version",
             *dataset_lines(),
             f"- Passages:  {payload['n_passages']:,}, one frozen {label} index searched by every row",
             f"- Seeds:     {', '.join(map(str, seeds))}; every trained model is trained once per seed",
             "- Scores:    nDCG@10 unless stated otherwise; pts = nDCG@10 x 100", "",
             *protocol_lines(seeds), "",
             "## 1. Informal queries: mean over seeds and stability", "",
             wrap("Sorted by mean informal nDCG@10. n = seeds scored. Gain = mean difference from base in pts. "
                  f"Gap closed = mean share of the base model's formal-informal gap recovered. {STD_READING}"), "",
             *table(["Model", "n", "Mean", "Gain (pts)", "Std (pts)", "Gap closed", "Sig. up/down"],
                    [[l, count(l), f"{agg(l)['informal']['nDCG@10']['mean']:.4f}", diff(l, "informal", "mean"),
                      diff(l, "informal", "std"), gap(l), sig(l, "informal")] for l in order]),
             "", "## 2. Formal queries: mean over seeds and stability", "",
             wrap("Did training hurt formal queries? Change = mean difference from base in pts; Std and "
                  "Sig. up/down as in section 1."), "",
             *table(["Model", "n", "Mean", "Change (pts)", "Std (pts)", "Sig. up/down"],
                    [[l, count(l), f"{agg(l)['formal']['nDCG@10']['mean']:.4f}", diff(l, "formal", "mean"),
                      diff(l, "formal", "std"), sig(l, "formal")] for l in order]),
             "", "## 3. Informal nDCG@10 per seed", "",
             "Range = best minus worst seed, in pts.", "", *per_seed("informal"),
             "", "## 4. Formal nDCG@10 per seed", "", *per_seed("formal")]
    for n, side in ((5, "informal"), (6, "formal")):
        lines += ["", f"## {n}. All metrics, {side} queries: mean (std) over seeds", "",
                  *table(["Model", *METRICS], [[l, *(mean_std(agg(l)[side][k]) for k in METRICS)] for l in order])]
    step = lambda l, s: "-" if not run(l, s) or run(l, s)["selected_step"] is None else str(run(l, s)["selected_step"])
    lines += ["", "## 7. Checkpoint step per seed", "",
              wrap("Optimiser step of the scored model: the best dev checkpoint unless the notes say otherwise. A "
                   "step that moves a lot between seeds means the dev selection itself is unstable."), "",
              *table(["Model", *map(str, seeds)], [[l, *(step(l, s) for s in seeds)] for l in trained])]
    row_notes = [wrap(f"- Seeds: {seed_note(report_name(payload['encoder']).replace('test', args.split) + '.md')}", "  ")]
    row_notes += [wrap(f"- {l}: {results[l]['note']}", "  ") for l in order if results[l]["note"]]
    if payload["encoder"] == "me5":
        row_notes.append(wrap(f"- Temperature: {TAU_NOTE}", "  "))
    lines += ["", "## Notes", "", *row_notes]
    if skipped:
        lines += ["", "## Missing runs", "", *missing_lines(skipped)]
    return "\n".join(lines) + "\n"


def std_compare_markdown(payloads: list, args) -> str:
    """Seed stability per method and encoder: mean effect and its std across seeds."""
    names = [SHORT.get(p["encoder"], p["encoder"]) for p in payloads]
    seeds = payloads[0]["seeds"]
    methods = [l for l, kind, _ in rows(payloads[0]["encoder"]) if kind != "base"]
    agg = lambda p, l: (p["results"].get(l) or {}).get("aggregate")

    def diff(p, l, side, key):
        a = agg(p, l)
        if a is None:
            return "n/a"
        d = a[f"{side}_vs_base_pts"]
        return "-" if d is None or d[key] is None else f"{d[key]:+.2f}" if key == "mean" else f"{d[key]:.2f}"

    def sig(p, l, side):
        a = agg(p, l)
        return "n/a" if a is None else "-" if p["results"][l]["reference"] else \
            "{up}/{down}".format(**a[f"{side}_significant"])

    def gap(p, l):
        g = (agg(p, l) or {}).get("gap_closed_pct")
        return "n/a" if g is None else f"{g['mean']:.1f}%" if g["std"] is None else f"{g['mean']:.1f}% ({g['std']:.1f})"

    effect = table(["Model", "Encoder", "Informal", "Std", "Sig.", "Formal", "Std", "Sig."],
                   by_model(methods, payloads, names, lambda p, l: (
                       diff(p, l, "informal", "mean"), diff(p, l, "informal", "std"), sig(p, l, "informal"),
                       diff(p, l, "formal", "mean"), diff(p, l, "formal", "std"), sig(p, l, "formal"))))
    short_runs = [f"{n} {l} (n = {agg(p, l)['n']})" for n, p in zip(names, payloads) for l in methods
                  if agg(p, l) and not p["results"][l]["reference"] and agg(p, l)["n"] < len(seeds)]
    lines = [f"# Seed stability across encoders: MIRACL-id {args.split} set", "",
             *[f"- {n + ':':<10} {p['encoder_label']} ({p['model_id']})" for n, p in zip(names, payloads)],
             f"- Queries:   {payloads[0]['n_queries']} {args.split} queries, each in a formal and an informal version",
             "- Passages:  each encoder searches its own frozen index of the same "
             f"{payloads[0]['n_passages']:,} passages",
             f"- Seeds:     {', '.join(map(str, seeds))}; every trained model is trained once per seed and encoder",
             "- Scores:    nDCG@10; pts = nDCG@10 x 100; every gain is measured against THAT encoder's own base", "",
             *protocol_lines(seeds), "",
             "## 1. Mean gain over each encoder's base and its std across seeds", "",
             wrap("Informal = mean informal gain, Formal = mean formal change, both in pts over seeds; Sig. = the "
                  "Sig. up/down count below. " + STD_READING), "", *effect,
             "", "## 2. Gap closed on informal queries: mean (std across seeds)", "",
             *table(["Model", *names], [[l, *(gap(p, l) for p in payloads)] for l in methods]),
             "", "## Notes", "",
             wrap("- Per encoder (scores per seed, MRR@10, Recall@100, checkpoint steps): "
                  + ", ".join(report_name(p["encoder"], std=True).replace("test", args.split) + ".md" for p in payloads)
                  + ".", "  "),
             wrap("- Seeds: " + seed_note(" / ".join(report_name(p["encoder"]).replace("test", args.split) + ".md"
                                                    for p in payloads)), "  ")]
    if short_runs:
        lines.append(wrap(f"- Fewer than {len(seeds)} seeds scored: {', '.join(short_runs)}.", "  "))
    lines += [*[wrap(f"- {l}: {note}", "  ") for l, note in notes(payloads[0]["encoder"], "base", "d").items()],
              wrap(f"- Adapter size: {DIM_NOTE}.", "  "),
              wrap(f"- Temperature: {TAU_NOTE}", "  ")]
    return "\n".join(lines) + "\n"


def grouped(skipped: dict) -> dict:
    """Missing rows with the seeds that share a reason on one line: {label: "seeds 42, 84: why; ..."}."""
    out = {}
    for label, why in skipped.items():
        if isinstance(why, str):
            out[label] = why
            continue
        by_reason = {}
        for seed, w in why.items():
            by_reason.setdefault(w, []).append(seed)
        out[label] = "; ".join(f"seed{'s' * (len(s) > 1)} {', '.join(s)}: {w}" for w, s in by_reason.items())
    return out


def testset_markdown(payload: dict, args) -> str:
    """Report of the extra test sets for one encoder; plain ASCII like markdown()."""
    enc, label, seeds = payload["encoder"], payload["encoder_label"], payload["seeds"]
    f4 = lambda x: f"{x:.4f}"
    ci = lambda b: f"[{100 * b['ci95_low']:+.2f}, {100 * b['ci95_high']:+.2f}]"
    pct_ci = lambda b: f"[{100 * b['ci95_low']:.1f}%, {100 * b['ci95_high']:.1f}%]"
    p_val = lambda b: f"{b['p']:.3f}{star(b)}"
    sig = lambda b: "*" if b["p"] < 0.05 else ""  # inline text: no padding
    in_pts = lambda b: f"{100 * b['diff']:+.2f}{sig(b)} pts {ci(b)}, p {b['p']:.3f}"
    mean_std = lambda s: f"{s['mean']:+.2f}" if s["std"] is None else f"{s['mean']:+.2f} ({s['std']:.2f})"
    updown = lambda c: "{up}/{down}".format(**c)
    scored = {l for ts in TESTSETS for l in (payload.get(ts) or {}).get("models", {})}
    lines = [f"# Extra test sets: {label}", "",
             f"- Encoder:   {label} ({payload['model_id']})",
             f"- Models:    the reported checkpoints, not retrained; every trained model once per seed "
             f"({', '.join(map(str, seeds))})",
             f"- Scores:    nDCG@10 unless stated otherwise; pts = nDCG@10 x 100; gains against {label} base on",
             f"             the same queries; * = p < 0.05 (bootstrap, {payload['bootstrap_samples']} samples)",
             "- Built by:  comparison/prepare_testset.py; scored by comparison/evaluate_all.py --testset"]
    skipped = {}

    sp = payload.get("style_pilot")
    if sp:
        b, n, models = sp["base"], sp["n_queries"], sp["models"]
        agg = lambda l: models[l]["aggregate"]
        order = sorted(models, key=lambda l: -agg(l)["varied_vs_base_pts"]["mean"])
        score_row = lambda name, side, vs: [name, *(f4(b["scores"][side][k]) for k in METRICS), vs]
        shift = lambda l: agg(l)["seed_averaged"]["varied_minus_template"]
        lines += ["", "## A. Style shift: do the gains hold for a different informal style?", "",
                  wrap(f"Data: the {n} test queries that also have a varied-style informal rewrite "
                       "(../BGE_M3/data/pairs/new_pairs.csv). Same questions, same formal text, two informal "
                       "versions: template = the style of every training query (train_pairs.csv: lower-case, "
                       "question word moved to the end, mostly '... sih'); varied = a rewrite whose style is sampled "
                       "per query (slang, abbreviations, typos, word order, capitalisation, particles)."), "",
                  "### A1. Base model on the three versions", "",
                  *table(["Version", *METRICS, "vs formal (pts)"],
                         [score_row("Formal", "formal", "-"),
                          score_row("Informal, template", "template", pts(b["template_vs_formal"])),
                          score_row("Informal, varied", "varied", pts(b["varied_vs_formal"]))]), "",
                  wrap(f"Varied minus template on base: {in_pts(b['varied_vs_template'])}. Negative = the varied "
                       f"style is harder for {label} than the template style."),
                  "", "### A2. Gain over base per informal style", "",
                  wrap("Gain = mean difference from base over seeds in pts (std across seeds in brackets). Sig. = "
                       "seeds significantly better / worse than base on their own (paired bootstrap over the "
                       "queries, p < 0.05). Gap closed T / V = share of base's formal-informal gap of that style "
                       "recovered, mean over seeds."), "",
                  *table(["Model", "n", "Template gain", "Sig.", "Varied gain", "Sig.", "Gap closed T", "Gap closed V"],
                         [[l, str(agg(l)["n"]), mean_std(agg(l)["template_vs_base_pts"]),
                           updown(agg(l)["template_significant"]), mean_std(agg(l)["varied_vs_base_pts"]),
                           updown(agg(l)["varied_significant"]),
                           f"{agg(l)['gap_closed_pct']['template']['mean']:.1f}%",
                           f"{agg(l)['gap_closed_pct']['varied']['mean']:.1f}%"] for l in order]),
                  "", "### A3. Does the gain shrink under the varied style?", "",
                  wrap("Change = varied gain minus template gain, in pts, per question. Change, 95% CI and p are for "
                       "the seed-averaged model (per-query scores averaged over seeds), paired bootstrap over the "
                       f"{n} questions; Seeds up/down = single seeds whose change is significant on its own. Change "
                       "near 0 = the gain carries over to the new style; clearly negative = part of the gain is tied "
                       "to the training template."), "",
                  *table(["Model", "Change (pts)", "95% CI", "p", "Seeds up/down"],
                         [[l, pts(shift(l)), ci(shift(l)), p_val(shift(l)),
                           updown(agg(l)["varied_minus_template_significant"])] for l in order])]
        skipped.update({f"{l} (style shift)": w for l, w in sp["skipped"].items()})

    dv = payload.get("miracl_dev_formal")
    if dv:
        encs, drop, models = dv["encoders"], dv["drop"], dv["models"]
        name = lambda e: ENCODERS[e].label + ("" if e == enc else " (yardstick)")
        agg = lambda l: models[l]["aggregate"]
        order = [l for l, _, _ in rows(enc) if l in models]
        lines += ["", "## B. Contamination: formal queries the encoder has seen vs never seen", "",
                  wrap(f"Seen = our {dv['n_seen']} test queries. They come from the official MIRACL-id train split, "
                       f"which {label} was fine-tuned on together with their relevant passages. Unseen = "
                       f"{dv['n_unseen']} official MIRACL-id dev queries (data/miracl_dev/, from "
                       "data/topics.miracl-v1.0-id-dev.tsv), never used to train the encoders. Formal queries only: "
                       "the dev queries have no informal version yet. Both sets search the same 500,000 passages."), "",
                  "### B1. Base models: drop from seen to unseen queries", "",
                  wrap("Drop = seen minus unseen nDCG@10, in pts; Relative = the drop as a share of the seen score. "
                       "95% CI and p from a bootstrap that resamples each query set on its own (the sets differ, "
                       "so it is unpaired)."), "",
                  *table(["Encoder", "Seen", "Unseen", "Drop (pts)", "95% CI", "Relative", "95% CI"],
                         [[name(e), f4(dv["base"][e]["seen"]["nDCG@10"]), f4(dv["base"][e]["unseen"]["nDCG@10"]),
                           pts(drop[e]["abs"]), ci(drop[e]["abs"]), f"{100 * drop[e]['rel']['diff']:.1f}%",
                           pct_ci(drop[e]["rel"])] for e in encs])]
        for e in encs[1:]:
            m = drop[f"{encs[0]}_minus_{e}"]
            lines += ["", wrap(f"Memory effect = {label} drop minus {ENCODERS[e].label} drop: {in_pts(m['abs'])}; "
                               f"relative: {100 * m['rel']['diff']:+.1f}{sig(m['rel'])} percentage points "
                               f"{pct_ci(m['rel'])}, p {m['rel']['p']:.3f}. {ENCODERS[e].label} was "
                               "never trained on MIRACL, so its drop shows how much harder the unseen queries are by "
                               f"themselves; the extra drop of {label} is what training on the seen queries added. "
                               f"Compare the relative column too: {ENCODERS[e].label} scores far lower, so the same "
                               "difficulty costs it fewer absolute points.")]
        lines += ["", "### B2. All metrics, base models", "",
                  *table(["Encoder", "Queries", *METRICS],
                         [[name(e) if w == "seen" else "", f"{w} ({dv['n_' + w]})",
                           *(f4(dv["base"][e][w][k]) for k in METRICS)] for e in encs for w in ("seen", "unseen")])]
        if models:
            lines += ["", "### B3. Trained models on formal queries: seen vs unseen", "",
                      wrap("Change = mean nDCG@10 difference from base over seeds in pts (std across seeds in "
                           "brackets), on the formal queries. Sig. = seeds significantly better / worse than base. "
                           "About 0 on both = formal retrieval stays intact; a change only on the unseen queries = "
                           "the preservation does not carry over to new questions."), "",
                      *table(["Model", "n", "Seen change", "Sig.", "Unseen change", "Sig."],
                             [[l, str(agg(l)["n"]), mean_std(agg(l)["seen_vs_base_pts"]),
                               updown(agg(l)["seen_significant"]), mean_std(agg(l)["unseen_vs_base_pts"]),
                               updown(agg(l)["unseen_significant"])] for l in order])]
        skipped.update({f"{l} (seen vs unseen)": w for l, w in dv["skipped"].items()})

    lines += ["", "## Notes", "",
              wrap("- Seeds: the same five training runs per model as report_std_test.md; nothing was retrained. "
                   "Base involves no training and is scored once.", "  "),
              *[wrap(f"- {l}: {note}", "  ") for l, note in notes(enc).items() if l in scored]]
    if skipped:
        lines += ["", "## Missing rows", "", *[wrap(f"- {l}: {why}", "  ") for l, why in grouped(skipped).items()]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
