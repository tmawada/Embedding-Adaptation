# Informal-query adaptation for BGE-M3 (MIRACL Indonesian)

Making dense retrieval robust to informal/slang Indonesian queries **without rebuilding the passage
index**: the document tower stays frozen, only the query side is adapted. Dense retrieval throughout.

## Layout

| Path | What it is |
|---|---|
| `model.py`, `dataset.py`, `eval.py` | Shared library: frozen BGE-M3 encoder, adapter / GRL / discriminator, data loading, metrics + paired bootstrap. Every method folder imports these from the project root. |
| `prepare.py` | Builds the base embedding cache: query/passage vectors, query splits, mined hard negatives. Run this first. |
| `query_dann/` | **Query-DANN v2**: residual adapter + gradient reversal + sanitized domain discriminator, with InfoNCE and a formal-preservation loss. Includes the inference demo and interactive search. |
| `query_distillation/` | **Query distillation**: teacher = the base encoder on the formal twin query; `embed` (match the vector) or `score` (match the ranking of the teacher's top-100). |
| `uda_query_dann/` | **Strict UDA**: labelled formal queries as source, unlabelled and unpaired informal queries as target (disjoint or overlapping domains, strict vs oracle checkpoint selection). |
| `tsdae_sapt/` | **TSDAE-SAPT** on informal Indonesian reviews (prdect-id): sentence-level adaptive pretraining of the query tower, plus its zero-shot evaluation. |
| `tsdae_sapt_queries/` | SAPT data ablation: the same objective trained on the task's own informal queries. |
| `llm_rewrite/` | **LLM normalization** baseline: Gemini rewrites the informal query, the untouched encoder embeds it. |
| `encoder_comparison/` | Second encoder (multilingual-e5-large-instruct) and cross-encoder tables. |
| `comparison/` | Harmonized retraining (identical protocol for every trained model) and the final reports. |

Not versioned (regenerable, see `.gitignore`): `data/`, `**/cache/`, `**/output/`, checkpoints, logs.
Small result summaries under `**/runs/` are kept.

## Quickstart

```bash
python3 prepare.py                  # embedding cache (~15 min on an RTX 3090, ~1 GB)
bash query_dann/run.sh              # Query-DANN v2: gamma sweep + test evaluation
bash query_distillation/run.sh      # distillation variants
bash uda_query_dann/run.sh          # strict UDA variants
bash tsdae_sapt/run.sh              # SAPT pretraining + zero-shot evaluation
bash comparison/run_harmonized.sh   # everything under one shared training protocol
python3 comparison/report.py        # final tables (formal vs informal, nDCG gap)
```

The LLM baseline needs a Google AI Studio key in `~/.gemini_api_key`:

```bash
python3 llm_rewrite/normalize.py --split test --model gemini-3.5-flash-lite --batch_size 40
```

## Shared training protocol (harmonized runs)

1350 optimiser steps · effective batch 64 · AdamW lr 3e-5 → 1e-6 cosine (10% warmup) · weight decay 0.01 ·
grad clip 1.0 · bf16 · max_length 128 · seed 42 · checkpoint = best dev nDCG@10 on informal queries
(evaluated every 50 steps). Method-specific knobs (gamma, mu, tau, bottleneck, del_ratio, temperature) keep
their own values.

## Evaluation

578 held-out MIRACL-id queries against 500,000 passages; Recall@100, MRR@100, nDCG@10, with 1000-sample
paired bootstrap tests. Results: `comparison/runs/harmonized/report_harmonized_test.md`.

Environment: Python 3.8, torch 2.1.2+cu121, transformers 4.45.2, one RTX 3090.
