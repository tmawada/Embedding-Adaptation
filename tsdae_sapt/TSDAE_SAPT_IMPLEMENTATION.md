# TSDAE-SAPT for BGE-M3 — design record and results

**Status: negative result. The line is retired.** Training code is `train_sapt.py` (authoritative for
the method); evaluation is `prepare_sapt_cache.py` + `evaluate_sapt.py`. This file previously held a
design sketch written *before* the code and superseded by it — see "Corrections" below for the parts
that were wrong.

## What this was trying to do

Adapt the **BGE-M3** query tower so it natively understands informal/slang Indonesian, by unsupervised
continued pretraining on informal product reviews — then measure what that does to MIRACL-id retrieval.

TSDAE (Wang et al., 2021): corrupt a sentence by deleting ~60% of its words, compress the remainder into
a single `[CLS]` vector, and train a decoder to reconstruct the original sentence from that one vector.
The bottleneck forces the encoder to model the *deleted* tokens, not just the visible ones — the hope
being that this transfers to guessing informal/slang surface forms from context.

Unlike the rest of the repo, this method **touches the encoder weights and rebuilds the index**. There
is no query adapter anywhere in this folder.

## Data (`data/prdect-id.csv`)

Indonesian product reviews. Fully unsupervised: the `Sentiment` label is discarded, only
`Customer Review` is used.

- 5,305 unique reviews → **4,911 kept** (deduplicated, non-empty, ≥ `min_words` = 4 words — short
  reviews can't survive 60% deletion).

## Method (as implemented)

```
noisy review (60% words deleted, resampled each draw)
  → BGE-M3 encoder  [FULLY FINE-TUNED]  → last_hidden_state[:, 0]  = [CLS] (1024-d)
      ↑                                        │  sole information channel, pre-normalisation
      │        gradients from reconstruction   ↓
      └────────── XLMRobertaForCausalLM decoder, cross-attends to that single vector
                  → token cross-entropy vs the ORIGINAL uncorrupted review
```

- **Encoder**: `BAAI/bge-m3`, 568M params, trained. Saving the encoder only is the deliverable; the
  decoder is discarded after training.
- **Decoder**: full `XLMRobertaForCausalLM` (not a shallow stack), weights **tied** to the encoder by
  parameter name — 389 tensors shared, 240 new (cross-attention + LM-head transform), plus the tied
  word-embedding matrix as `lm_head`. Tying means reconstruction gradients flow straight into the
  encoder.
- **Loss**: cross-entropy against the uncorrupted review, pads ignored.
- **Optimisation**: AdamW, lr 3e-5 constant, weight decay 0, grad clip 1.0, bf16 autocast, batch 8,
  max_length 128, seed 42.

Run: **614 steps / 1 epoch, reconstruction loss 16.77 → 6.28** (~5 min, RTX 3090). Model at
`output/bge-m3-sapt-informal-id/` (~2.3 GB) with a `sapt_config.json` sidecar. Log: `runs/train/log.jsonl`.

## Corrections to the original design sketch

The first version of this file described a configuration that would have produced no effect, and gave a
dependency rationale that does not hold:

1. **"Frozen encoder" was wrong.** A frozen encoder cannot change its embeddings, so the entire
   experiment would have been a no-op — the loss would still fall (the decoder learns to decode a fixed
   representation) while every embedding stayed bit-identical to base BGE-M3. The implementation fully
   fine-tunes the encoder. This is the one place the code is unambiguously better than the plan.
2. **The stated reason for bypassing sentence-transformers was wrong.** The code claims
   "sentence-transformers needs Python >= 3.10". It does not: sentence-transformers 3.2.1 imports
   cleanly on this Python 3.8.10, and `losses.DenoisingAutoEncoderLoss` provides
   `tie_encoder_decoder=True` exactly as hand-rolled here. The rewrite bought direct control of the
   `[CLS]`-only bottleneck (no L2 normalisation mid-training) and native bf16 autocast — a deliberate
   choice, not a hard blocker. The only genuine gap is that `nltk` is **not installed** here, which
   `DenoisingAutoEncoderDataset` needs for its punkt sentence splitting.
3. **Column name**: the sketch guessed `'Review'` / `'text'`; the column is `'Customer Review'`.

The sketch also contained no evaluation, no dual cache modes, no hard-negative re-mining, and no drift
diagnostics — those were added during implementation.

## Results

MIRACL-id, 578 test queries, 500,000 passages. nDCG@10. Three conditions, identical queries and splits
(`runs/zero_shot_test.json`):

| condition | queries | passages | formal | informal | drop | cos(inf, formal twin) |
|---|---|---|---|---|---|---|
| base BGE-M3 | base | base | 0.6166 | 0.5373 | −0.0793 | 0.8976 |
| `query_only` | SAPT | base (symlinked) | 0.5340 | 0.3604 | −0.1736 | 0.8085 |
| `symmetric` | SAPT | SAPT | 0.4289 | 0.2468 | −0.1820 | 0.8085 |

Embedding drift vs base (mean cosine to the same vector under base): formal queries 0.809, informal
0.761, passages 0.704 (and exactly 1.000 in `query_only`, confirming the symlink). Every paired
bootstrap against base is p ≈ 0.001 with CIs excluding zero. Dev reproduces test almost exactly
(0.6144 → 0.4124 formal, 0.5328 → 0.2342 informal).

**Reading the three rows:**

- The style gap *did* collapse — cos(informal, formal twin) fell 0.898 → 0.808, so the encoder really
  did merge the two styles. But it merged them by **degrading the informal side** (0.537 → 0.360), not
  by lifting it, and the formal side degraded too (0.617 → 0.534) despite never being in the training
  data. That's forgetting, not adaptation.
- Rebuilding the index didn't rescue it. `symmetric` re-encodes passages to meet the queries (drift
  only 0.704, so the index did move) and still scores *worse* than `query_only`. The failure is not a
  query/passage space mismatch — **the encoder's retrieval semantics were harmed directly.**

## Cross-experiment evidence (the conclusion does not hinge on this folder)

- The domain-mismatch explanation (prdect-id reviews ≠ MIRACL factoid queries) was tested and
  **refuted** by `../tsdae_sapt_queries/`, which runs the same objective on the task's *own* informal
  queries: 0.5266 informal vs 0.5373 base in the harmonized protocol (p = 0.001). Training on
  off-domain reviews is not the problem; the `[CLS]`-only reconstruction bottleneck is.
- Stacking the broken tower with a trained adapter contributes **nothing**:
  `../multi_seed/runs/seed_report_test.md` reports TSDAE(task queries) + Query-DANN linear at
  0.5774 ± 0.0018 vs Query-DANN linear alone at 0.5775 ± 0.0008 — a difference of **−0.01 ± 0.21,
  p = 0.981**. The adapter learns to route around the damaged tower.
- Rankings live in `../comparison/runs/harmonized/report_harmonized_test.md`. Briefly: SAPT (reviews)
  alone 0.5351 informal, i.e. −0.22 pts vs base with the gap essentially unchanged (−7.92 vs −7.93);
  LLM normalisation 0.6017, gap narrowed **+6.44** — the only method with real headroom.

## Conclusion

TSDAE sentence-adaptive pretraining does not help this task — alone, or stacked under an adapter. The
`[CLS]`-only denoising bottleneck erodes retrieval quality faster than it transfers informal-register
fluency. **Do not rerun this folder.** The finding stands as a negative result: adapting encoder weights
on unlabelled informal text is dominated by query-side adapters and by LLM query normalisation on this
benchmark.
