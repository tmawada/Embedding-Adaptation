# Seed stability across encoders: MIRACL-id test set

- BGE-M3:    BGE-M3 (BAAI/bge-m3)
- mE5:       mE5-large-instruct (intfloat/multilingual-e5-large-instruct)
- mpnet:     multilingual-mpnet (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- Queries:   553 test queries, each in a formal and an informal version
- Passages:  each encoder searches its own frozen index of the same 500,000 passages
- Seeds:     42, 84, 123, 256, 512; every trained model is trained once per seed and encoder
- Scores:    nDCG@10; pts = nDCG@10 x 100; every gain is measured against THAT encoder's own base

## Training protocol (every trained model)

- Steps:       1350 optimiser steps, effective batch 64
- Optimiser:   AdamW, lr 3e-05, weight decay 0.01, grad clip 1.0
- Schedule:    cosine down to 1e-06 after 10% linear warmup
- Precision:   bf16, max_length 128 (encoder fine-tuning)
- Seeds:       42, 84, 123, 256, 512 (one training run per seed and model)
- Checkpoint:  best dev informal nDCG@10, checked every 50 steps

## 1. Mean gain over each encoder's base and its std across seeds

Informal = mean informal gain, Formal = mean formal change, both in pts over seeds; Sig. = the Sig.
up/down count below. Std = sample standard deviation across seeds (n - 1), in pts; it is the same
for the score and for the gain, because the base row does not depend on the seed. Sig. up/down = how
many seeds are significantly better / worse than base on their own (paired bootstrap on nDCG@10, p <
0.05). A model is stable when its std is small next to its gain and all seeds agree in Sig. up/down.

| Model                                 | Encoder | Informal |  Std | Sig. | Formal |  Std | Sig. |
|---------------------------------------|--------:|---------:|-----:|-----:|-------:|-----:|-----:|
| LLM rewrite                           |  BGE-M3 |    +6.21 |    - |    - |  -2.15 |    - |    - |
|                                       |     mE5 |    +3.40 |    - |    - |  -1.28 |    - |    - |
|                                       |   mpnet |   +10.67 |    - |    - |  +1.31 |    - |    - |
| Query-DANN (bottleneck)               |  BGE-M3 |    +1.38 | 0.36 |  3/0 |  -0.20 | 0.44 |  0/0 |
|                                       |     mE5 |    +0.69 | 0.70 |  3/0 |  -0.47 | 0.12 |  0/0 |
|                                       |   mpnet |    +8.43 | 0.48 |  5/0 |  +5.31 | 0.67 |  5/0 |
| Query-DANN (linear)                   |  BGE-M3 |    +1.91 | 0.07 |  5/0 |  +0.00 | 0.06 |  0/0 |
|                                       |     mE5 |    +0.80 | 0.08 |  5/0 |  -0.13 | 0.07 |  0/0 |
|                                       |   mpnet |    +9.39 | 0.12 |  5/0 |  +7.48 | 0.24 |  5/0 |
| Query distillation score (bottleneck) |  BGE-M3 |    +3.83 | 0.26 |  5/0 |  +0.04 | 0.20 |  0/0 |
|                                       |     mE5 |    +1.46 | 0.09 |  5/0 |  -0.22 | 0.22 |  0/0 |
|                                       |   mpnet |    +3.33 | 0.32 |  5/0 |  -0.39 | 0.34 |  0/0 |
| Query distillation score (linear)     |  BGE-M3 |    +3.75 | 0.13 |  5/0 |  +0.08 | 0.09 |  0/0 |
|                                       |     mE5 |    +1.66 | 0.16 |  5/0 |  -0.39 | 0.06 |  0/0 |
|                                       |   mpnet |    +3.18 | 0.01 |  5/0 |  +0.31 | 0.04 |  0/0 |
| TSDAE-SAPT (task queries), final      |  BGE-M3 |   -10.95 | 1.62 |  0/5 | -11.15 | 1.46 |  0/5 |
|                                       |     mE5 |   -27.67 | 4.19 |  0/5 | -28.41 | 3.66 |  0/5 |
|                                       |   mpnet |    -4.51 | 0.17 |  0/5 |  -5.95 | 1.03 |  0/5 |
| Query InfoNCE                         |  BGE-M3 |    +1.35 | 0.64 |  1/0 |  -3.67 | 0.63 |  0/5 |
|                                       |     mE5 |    -2.02 | 0.40 |  0/3 |  -4.17 | 0.23 |  0/5 |
|                                       |   mpnet |   +18.42 | 0.27 |  5/0 | +10.93 | 0.14 |  5/0 |

## 2. Gap closed on informal queries: mean (std across seeds)

| Model                                 |         BGE-M3 |             mE5 |        mpnet |
|---------------------------------------|---------------:|----------------:|-------------:|
| LLM rewrite                           |          79.7% |           82.3% |       108.1% |
| Query-DANN (bottleneck)               |    17.7% (4.6) |    16.7% (17.1) |  85.4% (4.9) |
| Query-DANN (linear)                   |    24.6% (0.9) |     19.2% (1.8) |  95.2% (1.2) |
| Query distillation score (bottleneck) |    49.2% (3.3) |     35.4% (2.2) |  33.8% (3.2) |
| Query distillation score (linear)     |    48.1% (1.7) |     40.3% (3.8) |  32.2% (0.1) |
| TSDAE-SAPT (task queries), final      | -140.4% (20.7) | -669.8% (101.4) | -45.7% (1.7) |
| Query InfoNCE                         |    17.3% (8.2) |    -49.0% (9.8) | 186.6% (2.7) |

## Notes

- Per encoder (scores per seed, MRR@10, Recall@100, checkpoint steps): report_std_test.md,
  report_std_test_me5.md, report_std_test_mpnet.md.
- Seeds: Only the training seed changes between runs (initialisation, batch order, sampled hard
  negatives, TSDAE noise, dropout); queries, split, caches and protocol are identical. Seed 42 is
  the run in report_test.md / report_test_me5.md / report_test_mpnet.md. Base and LLM rewrite
  involve no training: scored once, so they have no std.
- LLM rewrite: gemini-3.5-flash-lite, cached rewrites of the test queries (no API calls); not
  trained. Its formal column rewrites the formal queries too, since a deployed rewriter sees every
  query.
- Query-DANN (bottleneck): the default adapter: zero-initialised residual bottleneck d -> 128 -> d
  on the frozen base query vector. Same adapter in "Query distillation score (bottleneck)".
- Query-DANN (linear): --adapter linear: a single d x d linear map initialised to the identity. Same
  adapter in "Query distillation score (linear)".
- Query InfoNCE: --objective infonce: the full encoder fine-tuned with InfoNCE on informal queries
  against the frozen passage index, without the TSDAE reconstruction term that the method adds by
  default (weight 0.1).
- TSDAE-SAPT (task queries), final: the encoder after the last training step, NOT the best dev
  checkpoint like the other rows: dev selection keeps step 50, which has barely trained and scores
  like base. This row shows what TSDAE training itself does to retrieval.
- Adapter size: d = the encoder's output size (BGE-M3 1024, mE5-large-instruct 1024,
  multilingual-mpnet 768).
- Temperature: mE5-large-instruct's cosine scores are compressed (relevant and top-ranked
  non-relevant passages both score ~0.89, random ones ~0.76), while the InfoNCE / distillation
  temperature (tau 0.05) was set for BGE-M3. The protocol is kept identical on purpose, so mE5 may
  be at a disadvantage on those methods.
