"""Apple-to-apple comparison: train every method under ONE shared protocol, then score them all on the same test split.

Trained models (6):
  Query-DANN                 bottleneck adapter (default) and linear adapter
  Query distillation (score) bottleneck adapter (default) and linear adapter
  TSDAE-SAPT (task queries)  full encoder; reported at its LAST step (dev selection keeps a barely trained step 50)
  Query InfoNCE              full encoder, --objective infonce (InfoNCE only, no TSDAE reconstruction term)
Not trained, reported as references: the base encoder and the LLM rewrite (cached rewrites, no API calls).

Encoder modes (--encoder, methods/encoders.py):
  bge_m3   BAAI/bge-m3 (default; CLS pooling)
  me5      intfloat/multilingual-e5-large-instruct (mean pooling, instruction prefix on queries)
  mpnet    sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (mean pooling, 768-d, 128 tokens)
  arctic   Snowflake/snowflake-arctic-embed-l-v2.0 (CLS pooling, "query: " prefix on queries)
  compare  every encoder x the six models, then the side-by-side report (report_encoders_test.md)
Each encoder has its own frozen 500k-passage index (bge_m3: methods/cache, built by methods/prepare.py; the others:
methods/cache/<encoder cache>, built by methods/prepare_encoder.py), and every row of that encoder searches it.

Seed stability (--std true): every trained model is trained once per seed (--seeds, any of 42 84 123 256 512; default
all five) and the report gives the mean and standard deviation across seeds (report_std_test[_<encoder>].md; compare
mode: report_encoders_std_test.md). --std false (default) is the single-seed comparison above. Only the seed changes:
same split, caches and protocol. Seed 42 is the protocol seed, so its runs are the existing ones; other seeds get a
_seed<N> suffix (harmonized_seed84, harmonized_linear_me5_seed84, ...). To fit on disk, the 2.2 GB encoder weights of
a TSDAE / InfoNCE seed run are deleted once its query cache is built (configs and dev logs stay); --keep_seed_models
keeps them.

Shared protocol (every trained model; method-specific knobs such as gamma, kd weight or del_ratio keep their defaults):
  1350 optimiser steps | effective batch 64 | AdamW lr 3e-5 -> 1e-6, cosine with 10% linear warmup |
  weight decay 0.01 | grad clip 1.0 | bf16 | max_length 128 (encoder fine-tuning) | seed 42 |
  checkpoint = best dev informal nDCG@10, evaluated every 50 optimiser steps

Training artefacts go to methods/<method>/outputs/harmonized*/ (bge_m3) or harmonized*_<encoder>/; logs, status
and the reports to comparison/outputs/ (other encoders: logs/<encoder>/, status_<encoder>.json). Finished steps are
skipped on a re-run, so an interrupted run resumes where it stopped; evaluation re-runs whenever another step ran.

Dataset layouts (--dataset): miracl_train (default) = our 2,891 / 372 / 553 split of official MIRACL-id train, every
report above; official = data/official (preprocessing/prepare_official.py): train on official MIRACL-id train (3,444),
select on our dev split, test on the official MIRACL-id dev queries (867, informal in two styles). It gets its own cache
(methods/cache/official; other encoders: <encoder cache>/official, linking the same passage index), run names
(harmonized_official*, harmonized_me5_official*), status[_<encoder>]_official.json, logs/[<encoder>/]official/ and
reports (report_std_test[_<encoder>]_official.md: template style; ..._varied.md: varied style). Run --encoder bge_m3
first: the other encoders copy its split and query order.

  python3 comparison/run_all.py                         # BGE-M3 mode (~20 min on one RTX 3090)
  python3 comparison/run_all.py --encoder me5           # mE5 mode (~1 h, mostly encoding the passages once)
  python3 comparison/run_all.py --encoder compare       # both encoders, then the side-by-side evaluation
  python3 comparison/run_all.py --std true              # BGE-M3, five seeds (~15 min per extra seed)
  python3 comparison/run_all.py --encoder compare --std true --seeds 42 84 123
  python3 comparison/run_all.py --dataset official --std true   # BGE-M3 on data/official, five seeds
  python3 comparison/run_all.py --dataset official --std true --encoder mpnet   # then mpnet (or me5, arctic)
  python3 comparison/run_all.py --dry_run               # print the commands only
  python3 comparison/run_all.py --only query_dann evaluate
  python3 comparison/run_all.py --force                 # re-run finished steps too
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "outputs")
sys.path.insert(0, ROOT)

from methods.encoders import ENCODERS  # noqa: E402

PROTOCOL = {
    "total_steps": 1350, "effective_batch": 64, "optimizer": "AdamW", "lr": 3e-5, "lr_min": 1e-6,
    "scheduler": "cosine with linear warmup", "warmup_ratio": 0.1, "weight_decay": 0.01, "grad_clip": 1.0,
    "precision": "bf16", "max_length": 128, "seed": 42, "eval_every": 50, "selection": "best dev informal nDCG@10",
}
P = PROTOCOL
SEEDS = (42, 84, 123, 256, 512)  # --seeds choices for --std true; P["seed"] is the single-seed run

DATASETS = ("miracl_train", "official")  # --dataset; see the module docstring
DATASET = "miracl_train"


def set_dataset(name: str):
    """Select the dataset layout every path below follows (evaluate_all.py calls this too)."""
    global DATASET
    DATASET = name


def dtag() -> str:
    """'' for miracl_train (the existing names), _<dataset> otherwise."""
    return "" if DATASET == "miracl_train" else f"_{DATASET}"


def data_dir() -> str:
    """Relative to ROOT like every step path: data, or data/<dataset>."""
    return "data" if DATASET == "miracl_train" else os.path.join("data", DATASET)


def cache_dir(enc: str) -> str:
    """The encoder's cache (absolute), or its <dataset> subdirectory, which links the same passage index."""
    base = ENCODERS[enc].cache_dir
    return base if DATASET == "miracl_train" else os.path.join(base, DATASET)


def shared(seed: int) -> list:
    return ["--total_steps", P["total_steps"], "--lr", P["lr"], "--lr_min", P["lr_min"],
            "--warmup_ratio", P["warmup_ratio"], "--weight_decay", P["weight_decay"], "--eval_every", P["eval_every"],
            "--seed", seed]


def adapter_args(seed: int) -> list:
    """Adapters train on cached embeddings: the whole effective batch fits at once."""
    return shared(seed) + ["--batch_size", P["effective_batch"], "--grad_clip", P["grad_clip"],
                           "--mixed_precision", P["precision"]]


def encoder_args(seed: int) -> list:
    """Full-encoder fine-tuning: bf16 autocast and grad clip 1.0 are built into these trainers."""
    return shared(seed) + ["--scheduler", "cosine"]


def m(*parts: str) -> str:
    """Path under methods/, relative to ROOT (every step runs with cwd=ROOT)."""
    return os.path.join("methods", *parts)


def seed_tag(seed) -> str:
    """'' for the protocol seed (the original runs), _seed<N> otherwise."""
    return "" if seed in (None, P["seed"]) else f"_seed{seed}"


def run_names(enc: str, seed: int = P["seed"]):
    """bge_m3 keeps the original run names; other encoders get a suffix, e.g. harmonized_me5; other seeds too."""
    s = ("" if enc == "bge_m3" else f"_{enc}") + dtag() + seed_tag(seed)
    # harmonized_linear: --adapter linear; harmonized_infonce_only: --objective infonce
    return f"harmonized{s}", f"harmonized_linear{s}", f"harmonized_infonce_only{s}"


def artifacts(enc: str, seed: int = P["seed"]) -> dict:
    H, HL, HI = run_names(enc, seed)
    return {
        "cache": os.path.relpath(cache_dir(enc), ROOT),
        "query_dann": m("query_dann", "outputs", H, "best"),
        "query_dann_linear": m("query_dann", "outputs", HL, "best"),
        "query_distillation": m("query_distillation", "outputs", H, "best"),
        "query_distillation_linear": m("query_distillation", "outputs", HL, "best"),
        "tsdae_sapt_queries_final": m("tsdae_sapt_queries", "outputs", H, "model_final"),
        "tsdae_sapt_queries_final_cache": m("tsdae_sapt_queries", "outputs", H, "cache_final"),
        "query_infonce_model": m("query_infonce", "outputs", HI, "model"),
        "query_infonce_cache": m("query_infonce", "outputs", HI, "cache"),
    }


# Once a seed run's query cache exists, evaluation no longer needs the fine-tuned encoder: cache step -> model artefact.
PRUNE = {"tsdae_sapt_queries_final_cache": "tsdae_sapt_queries_final", "query_infonce_cache": "query_infonce_model"}


def report_name(enc: str, std: bool = False) -> str:
    name = "report_std_test" if std else "report_test"
    return (name if enc == "bge_m3" else f"{name}_{enc}") + dtag()


PY = [sys.executable, "-u"]

# A step is (encoder, seed or None, status key, name, command, file that exists once the step has produced its
# result). The key names the status entry and the log: the step name, plus _seed<N> for a non-protocol seed.


def cache_step(enc: str) -> tuple:
    if DATASET == "miracl_train":
        cmd = [*PY, m("prepare.py")] if enc == "bge_m3" else [*PY, m("prepare_encoder.py"), "--encoder", enc]
    elif enc == "bge_m3":  # the dataset's fixed split; passages linked from the encoder's cache, not re-encoded
        cmd = [*PY, m("prepare.py"), "--data_dir", data_dir(), "--cache_dir", artifacts(enc)["cache"],
               "--splits_file", os.path.join(data_dir(), "splits.json"),
               "--corpus_from", os.path.relpath(ENCODERS[enc].cache_dir, ROOT)]
    else:  # split and query order of BGE-M3's cache for this dataset, so it has to be built first
        if not os.path.exists(os.path.join(cache_dir("bge_m3"), "splits.json")):
            raise SystemExit(f"--dataset {DATASET}: run --encoder bge_m3 first (its cache sets the split)")
        cmd = [*PY, m("prepare_encoder.py"), "--encoder", enc, "--data_dir", data_dir(),
               "--cache_dir", artifacts(enc)["cache"], "--reference_cache", artifacts("bge_m3")["cache"],
               "--corpus_from", os.path.relpath(ENCODERS[enc].cache_dir, ROOT)]
    return (enc, None, "cache", "cache", cmd, os.path.join(artifacts(enc)["cache"], "hard_negs.json"))


def model_steps(enc: str, seed: int = P["seed"]) -> list:
    """Training and query-cache steps of the six models for one seed, in dependency order."""
    spec, A = ENCODERS[enc], artifacts(enc, seed)
    H, HL, _ = run_names(enc, seed)
    other = DATASET != "miracl_train"
    data = ["--cache_dir", A["cache"], *(["--data_dir", data_dir()] if other else [])]
    base = ["--data_dir", data_dir(), "--base_cache", A["cache"]] if other else []  # query-cache builders
    mode = ["--encoder", enc]
    adapter, encoder = adapter_args(seed), encoder_args(seed)
    out = [
        ("query_dann", [*PY, m("query_dann", "train.py"), "--output_dir", m("query_dann", "outputs"), "--run_name", H,
                        *data, *adapter], os.path.join(A["query_dann"], "config.json")),
        ("query_dann_linear", [*PY, m("query_dann", "train.py"), "--output_dir", m("query_dann", "outputs"),
                               "--run_name", HL, "--adapter", "linear", *data, *adapter],
         os.path.join(A["query_dann_linear"], "config.json")),
        ("query_distillation", [*PY, m("query_distillation", "train.py"), "--output_dir", m("query_distillation", "outputs"),
                                "--run_name", H, *data, *adapter], os.path.join(A["query_distillation"], "config.json")),
        ("query_distillation_linear", [*PY, m("query_distillation", "train.py"),
                                       "--output_dir", m("query_distillation", "outputs"), "--run_name", HL,
                                       "--adapter", "linear", *data, *adapter],
         os.path.join(A["query_distillation_linear"], "config.json")),
        # TSDAE is reported at its last step (--final_dir): dev selection keeps step 50, which has barely trained,
        # so the dev-selected encoder is not written (--save_best false).
        ("tsdae_sapt_queries", [*PY, m("tsdae_sapt_queries", "tsdae_train.py"),
                                "--output_dir", m("tsdae_sapt_queries", "outputs"), "--run_name", H,
                                "--model", spec.model_id, *mode, *data, "--batch_size", 64, "--grad_accum", 1,
                                "--max_seq_length", P["max_length"], "--save_best", "false",
                                "--final_dir", A["tsdae_sapt_queries_final"], *encoder],
         os.path.join(A["tsdae_sapt_queries_final"], "tsdae_info.json")),
        ("tsdae_sapt_queries_final_cache", [*PY, m("tsdae_sapt_queries", "prepare_tsdae_cache.py"), *mode,
                                            "--model", A["tsdae_sapt_queries_final"],
                                            "--out_cache", A["tsdae_sapt_queries_final_cache"], *base],
         os.path.join(A["tsdae_sapt_queries_final_cache"], "informal_emb.npy")),
        ("query_infonce", [*PY, m("query_infonce", "train_infonce.py"), "--output_dir", A["query_infonce_model"],
                           "--model", spec.model_id, *mode, *data, "--objective", "infonce", "--batch_size", 64,
                           "--max_length", P["max_length"], *encoder],
         os.path.join(A["query_infonce_model"], "infonce_config.json")),
        ("query_infonce_cache", [*PY, m("query_infonce", "prepare_infonce_cache.py"), *mode,
                                 "--model_path", A["query_infonce_model"], "--out_cache", A["query_infonce_cache"],
                                 *base],
         os.path.join(A["query_infonce_cache"], "informal_emb.npy")),
    ]
    return [(enc, seed, name + seed_tag(seed), name, cmd, marker) for name, cmd, marker in out]


def evaluate_step(enc: str, seeds=None) -> tuple:
    """enc = one encoder or "compare"; seeds = None for the single-seed report, a list for the seed-stability one."""
    std = ["--std", "true", "--seeds", *seeds] if seeds else []
    std += ["--dataset", DATASET] if DATASET != "miracl_train" else []
    if enc == "compare":
        report = "report_encoders_std_test" if seeds else "report_encoders_test"
    else:
        report = report_name(enc, std=bool(seeds))
    return (enc, None, "evaluate_std" if seeds else "evaluate", "evaluate",
            [*PY, os.path.join("comparison", "evaluate_all.py"), "--encoder", enc, *std],
            os.path.join("comparison", "outputs", report + ".json"))


def plan(mode: str, seeds=None) -> list:
    """Steps of one encoder mode, or of every encoder plus one joint evaluation; seeds=None trains each model once at
    the protocol seed, a list trains each model once per seed and ends with the seed-stability report."""
    encs = list(ENCODERS) if mode == "compare" else [mode]
    out = []
    for enc in encs:
        out.append(cache_step(enc))
        for seed in seeds or [P["seed"]]:
            out += model_steps(enc, seed)
        if mode != "compare":
            out.append(evaluate_step(enc, seeds))
    if mode == "compare":
        out.append(evaluate_step("compare", seeds))
    return out


def status_path(enc: str) -> str:
    return os.path.join(OUT, ("status" if enc == "bge_m3" else f"status_{enc}") + dtag() + ".json")


def log_path(enc: str, key: str) -> str:
    sub = ([] if enc == "bge_m3" else [enc]) + ([DATASET] if dtag() else [])
    return os.path.join(OUT, "logs", *sub, f"{key}.log")


def load_status(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def prune_weights(model_dir: str) -> float:
    """Delete a fine-tuned encoder's weight files, keeping its configs, selection info and dev log; returns GB freed."""
    freed = 0
    for fn in os.listdir(model_dir):
        if fn.endswith((".safetensors", ".bin")):
            path = os.path.join(model_dir, fn)
            freed += os.path.getsize(path)
            os.remove(path)
    return freed / 1e9


def main():
    names = sorted({s[3] for mode in (*ENCODERS, "compare") for s in plan(mode)})
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="bge_m3", choices=[*ENCODERS, "compare"],
                    help="Encoder mode: one encoder, or compare = every encoder x the six models + a joint report")
    ap.add_argument("--std", default="false", choices=["true", "false"],
                    help="Seed stability: train every model once per --seeds seed and report mean / std across seeds")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, choices=SEEDS, metavar="SEED",
                    help=f"Seeds for --std true, any of {' '.join(map(str, SEEDS))} (default: all five)")
    ap.add_argument("--keep_seed_models", action="store_true",
                    help="--std true: keep the 2.2 GB encoder weights of TSDAE / InfoNCE seed runs, which are deleted "
                         "by default once their query cache is built")
    ap.add_argument("--only", nargs="*", default=None, choices=names, help="Run just these steps (every seed)")
    ap.add_argument("--force", action="store_true", help="Re-run steps that already finished")
    ap.add_argument("--dry_run", action="store_true", help="Print the commands without running them")
    ap.add_argument("--dataset", default="miracl_train", choices=DATASETS,
                    help="Dataset layout: miracl_train (our split, every report so far) or official (data/official)")
    args = ap.parse_args()
    set_dataset(args.dataset)
    seeds = None
    if args.std == "true":
        seeds = sorted(set(args.seeds or SEEDS))
        if len(seeds) < 2:
            ap.error("--std true needs at least two seeds")
    elif args.seeds:
        ap.error("--seeds needs --std true")

    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    with open(os.path.join(OUT, "protocol.json"), "w") as f:
        json.dump(PROTOCOL, f, indent=2)

    todo = [s for s in plan(args.encoder, seeds) if args.only is None or s[3] in args.only]
    statuses = {}
    # Fewer fragmentation OOMs: query_infonce at batch 64 peaks at ~21.9 GB of the 24 GB card.
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")}
    t_all, ran = time.time(), False
    for i, (enc, seed, key, name, cmd, marker) in enumerate(todo, 1):
        cmd = [str(c) for c in cmd]
        shown = " ".join(cmd[2:])  # without the interpreter
        tag = f"[{i}/{len(todo)}] {enc}/{key}"
        status = statuses.setdefault(enc, load_status(status_path(enc)))
        stale = name == "evaluate" and ran  # the table must include whatever was just trained
        if not args.force and not stale and status.get(key, {}).get("done") and os.path.exists(os.path.join(ROOT, marker)):
            print(f"{tag}: done earlier, skipping")
            continue
        print(f"{tag}: python3 {shown}", flush=True)
        ran = True
        if args.dry_run:
            continue
        log = log_path(enc, key)
        os.makedirs(os.path.dirname(log), exist_ok=True)
        t0 = time.time()
        with open(log, "w") as f:
            f.write(f"$ python3 {shown}\n\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        minutes = (time.time() - t0) / 60
        ok = rc == 0 and os.path.exists(os.path.join(ROOT, marker))
        status[key] = {"done": ok, "minutes": round(minutes, 1), "exit_code": rc,
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "log": os.path.relpath(log, ROOT)}
        with open(status_path(enc), "w") as f:
            json.dump(status, f, indent=2)
        if not ok:
            with open(log) as f:
                tail = f.read().splitlines()[-25:]
            print(f"    FAILED after {minutes:.1f} min (exit {rc}, expected {marker}); last lines of "
                  f"{os.path.relpath(log, ROOT)}:\n      " + "\n      ".join(tail))
            sys.exit(1)
        print(f"    done in {minutes:.1f} min", flush=True)
        if seed_tag(seed) and name in PRUNE and not args.keep_seed_models:
            model_dir = artifacts(enc, seed)[PRUNE[name]]
            print(f"    deleted {prune_weights(os.path.join(ROOT, model_dir)):.1f} GB of encoder weights in "
                  f"{model_dir} (its query cache is built)", flush=True)

    if not args.dry_run:
        report = os.path.splitext(evaluate_step(args.encoder, seeds)[5])[0]
        print(f"all steps finished in {(time.time() - t_all) / 3600:.2f} h -> {report}.md")


if __name__ == "__main__":
    main()
