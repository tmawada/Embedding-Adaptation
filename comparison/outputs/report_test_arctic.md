# Harmonized comparison: MIRACL-id test set

- Encoder:   Arctic-Embed-L-2.0 (Snowflake/snowflake-arctic-embed-l-v2.0)
- Queries:   553 test queries, each in a formal and an informal version
- Passages:  500,000, one frozen Arctic-Embed-L-2.0 index searched by every row
- Scores:    nDCG@10 unless stated otherwise; pts = nDCG@10 x 100

## Training protocol (every trained model)

- Steps:       1350 optimiser steps, effective batch 64
- Optimiser:   AdamW, lr 3e-05, weight decay 0.01, grad clip 1.0
- Schedule:    cosine down to 1e-06 after 10% linear warmup
- Precision:   bf16, max_length 128 (encoder fine-tuning)
- Seed:        42
- Checkpoint:  best dev informal nDCG@10, checked every 50 steps

## 1. Summary

Sorted by informal nDCG@10. Gain and Change = difference from base in pts (* = p < 0.05, paired
bootstrap, 1000 samples). Gap closed = share of the base model's formal-informal gap recovered on
informal queries.

| Model                                 | Informal | Gain (pts) | Formal | Change (pts) | Gap closed |
|---------------------------------------|---------:|-----------:|-------:|-------------:|-----------:|
| Arctic-Embed-L-2.0 base (zero-shot)   |   0.4936 |         -  | 0.5174 |           -  |          - |
| Query InfoNCE                         |   0.5387 |     +4.51* | 0.5438 |       +2.64* |     190.0% |
| Query-DANN (linear)                   |   0.5239 |     +3.02* | 0.5354 |       +1.81* |     127.5% |
| Query-DANN (bottleneck)               |   0.5194 |     +2.58* | 0.5236 |       +0.63  |     108.7% |
| Query distillation score (linear)     |   0.5097 |     +1.60* | 0.5095 |       -0.79* |      67.5% |
| LLM rewrite                           |   0.5094 |     +1.58* | 0.5062 |       -1.11* |      66.5% |
| Query distillation score (bottleneck) |   0.5042 |     +1.05* | 0.5112 |       -0.62* |      44.4% |
| TSDAE-SAPT (task queries), final      |   0.4680 |     -2.56* | 0.4774 |       -4.00* |    -108.2% |

## 2. All metrics, informal queries

| Model                                 | nDCG@10 | MRR@10 | Recall@100 |
|---------------------------------------|--------:|-------:|-----------:|
| Arctic-Embed-L-2.0 base (zero-shot)   |  0.4936 | 0.6105 |     0.8469 |
| Query InfoNCE                         |  0.5387 | 0.6555 |     0.8798 |
| Query-DANN (linear)                   |  0.5239 | 0.6354 |     0.8802 |
| Query-DANN (bottleneck)               |  0.5194 | 0.6342 |     0.8757 |
| Query distillation score (linear)     |  0.5097 | 0.6316 |     0.8551 |
| LLM rewrite                           |  0.5094 | 0.6302 |     0.8614 |
| Query distillation score (bottleneck) |  0.5042 | 0.6220 |     0.8571 |
| TSDAE-SAPT (task queries), final      |  0.4680 | 0.5814 |     0.8231 |

## 3. All metrics, formal queries

| Model                                 | nDCG@10 | MRR@10 | Recall@100 |
|---------------------------------------|--------:|-------:|-----------:|
| Arctic-Embed-L-2.0 base (zero-shot)   |  0.5174 | 0.6355 |     0.8676 |
| Query InfoNCE                         |  0.5438 | 0.6504 |     0.8916 |
| Query-DANN (linear)                   |  0.5354 | 0.6540 |     0.8969 |
| Query-DANN (bottleneck)               |  0.5236 | 0.6495 |     0.8820 |
| Query distillation score (linear)     |  0.5095 | 0.6283 |     0.8658 |
| LLM rewrite                           |  0.5062 | 0.6253 |     0.8551 |
| Query distillation score (bottleneck) |  0.5112 | 0.6303 |     0.8606 |
| TSDAE-SAPT (task queries), final      |  0.4774 | 0.5966 |     0.8223 |

## 4. Significance vs base and checkpoint step

Informal / Formal = nDCG@10 difference from base in pts, paired bootstrap with 1000 samples; * = p <
0.05. Step = optimiser step of the scored model: the best dev checkpoint unless the notes say
otherwise (- = not trained).

| Model                                 | Informal |         95% CI |      p | Formal |      p | Step |
|---------------------------------------|---------:|---------------:|-------:|-------:|-------:|-----:|
| Query InfoNCE                         |   +4.51* | [+2.33, +6.52] | 0.001* | +2.64* | 0.002* | 1000 |
| Query-DANN (linear)                   |   +3.02* | [+1.88, +4.12] | 0.001* | +1.81* | 0.001* |  550 |
| Query-DANN (bottleneck)               |   +2.58* | [+1.05, +4.01] | 0.001* | +0.63  | 0.207  | 1350 |
| Query distillation score (linear)     |   +1.60* | [+0.61, +2.59] | 0.002* | -0.79* | 0.007* | 1150 |
| LLM rewrite                           |   +1.58* | [+0.36, +2.84] | 0.013* | -1.11* | 0.016* |    - |
| Query distillation score (bottleneck) |   +1.05* | [+0.09, +2.14] | 0.044* | -0.62* | 0.032* |  450 |
| TSDAE-SAPT (task queries), final      |   -2.56* | [-4.19, -0.81] | 0.008* | -4.00* | 0.001* | 1350 |

## Notes

- Query InfoNCE: --objective infonce: the full encoder fine-tuned with InfoNCE on informal queries
  against the frozen passage index, without the TSDAE reconstruction term that the method adds by
  default (weight 0.1).
- Query-DANN (linear): --adapter linear: a single 1024 x 1024 linear map initialised to the
  identity. Same adapter in "Query distillation score (linear)".
- Query-DANN (bottleneck): the default adapter: zero-initialised residual bottleneck 1024 -> 128 ->
  1024 on the frozen Arctic-Embed-L-2.0 query vector. Same adapter in "Query distillation score
  (bottleneck)".
- LLM rewrite: gemini-3.5-flash-lite, cached rewrites of the test queries (no API calls); not
  trained. Its formal column rewrites the formal queries too, since a deployed rewriter sees every
  query.
- TSDAE-SAPT (task queries), final: the encoder after the last training step, NOT the best dev
  checkpoint like the other rows: dev selection keeps step 50, which has barely trained and scores
  like base. This row shows what TSDAE training itself does to retrieval.
