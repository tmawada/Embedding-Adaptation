# Harmonized comparison: MIRACL-id test set

- Encoder:   BGE-M3 (BAAI/bge-m3)
- Queries:   553 test queries, each in a formal and an informal version
- Passages:  500,000, one frozen BGE-M3 index searched by every row
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
| BGE-M3 base (zero-shot)               |   0.5362 |         -  | 0.6141 |           -  |          - |
| LLM rewrite                           |   0.5983 |     +6.21* | 0.5926 |       -2.15* |      79.7% |
| Query distillation score (linear)     |   0.5725 |     +3.64* | 0.6146 |       +0.05  |      46.7% |
| Query distillation score (bottleneck) |   0.5710 |     +3.49* | 0.6153 |       +0.12  |      44.8% |
| Query-DANN (linear)                   |   0.5549 |     +1.88* | 0.6135 |       -0.05  |      24.1% |
| Query-DANN (bottleneck)               |   0.5488 |     +1.26  | 0.6132 |       -0.09  |      16.2% |
| Query InfoNCE                         |   0.5465 |     +1.03  | 0.5775 |       -3.65* |      13.2% |
| TSDAE-SAPT (task queries), final      |   0.4446 |     -9.16* | 0.5247 |       -8.94* |    -117.5% |

## 2. All metrics, informal queries

| Model                                 | nDCG@10 | MRR@10 | Recall@100 |
|---------------------------------------|--------:|-------:|-----------:|
| BGE-M3 base (zero-shot)               |  0.5362 | 0.6364 |     0.8986 |
| LLM rewrite                           |  0.5983 | 0.7042 |     0.9350 |
| Query distillation score (linear)     |  0.5725 | 0.6761 |     0.9136 |
| Query distillation score (bottleneck) |  0.5710 | 0.6722 |     0.9098 |
| Query-DANN (linear)                   |  0.5549 | 0.6589 |     0.9068 |
| Query-DANN (bottleneck)               |  0.5488 | 0.6554 |     0.8966 |
| Query InfoNCE                         |  0.5465 | 0.6412 |     0.9018 |
| TSDAE-SAPT (task queries), final      |  0.4446 | 0.5439 |     0.8174 |

## 3. All metrics, formal queries

| Model                                 | nDCG@10 | MRR@10 | Recall@100 |
|---------------------------------------|--------:|-------:|-----------:|
| BGE-M3 base (zero-shot)               |  0.6141 | 0.7196 |     0.9375 |
| LLM rewrite                           |  0.5926 | 0.6987 |     0.9304 |
| Query distillation score (linear)     |  0.6146 | 0.7192 |     0.9404 |
| Query distillation score (bottleneck) |  0.6153 | 0.7180 |     0.9400 |
| Query-DANN (linear)                   |  0.6135 | 0.7180 |     0.9369 |
| Query-DANN (bottleneck)               |  0.6132 | 0.7212 |     0.9394 |
| Query InfoNCE                         |  0.5775 | 0.6732 |     0.9241 |
| TSDAE-SAPT (task queries), final      |  0.5247 | 0.6352 |     0.8735 |

## 4. Significance vs base and checkpoint step

Informal / Formal = nDCG@10 difference from base in pts, paired bootstrap with 1000 samples; * = p <
0.05. Step = optimiser step of the scored model: the best dev checkpoint unless the notes say
otherwise (- = not trained).

| Model                                 | Informal |          95% CI |      p | Formal |      p | Step |
|---------------------------------------|---------:|----------------:|-------:|-------:|-------:|-----:|
| LLM rewrite                           |   +6.21* |  [+4.69, +7.87] | 0.001* | -2.15* | 0.001* |    - |
| Query distillation score (linear)     |   +3.64* |  [+2.64, +4.74] | 0.001* | +0.05  | 0.846  |  450 |
| Query distillation score (bottleneck) |   +3.49* |  [+2.50, +4.53] | 0.001* | +0.12  | 0.608  |  950 |
| Query-DANN (linear)                   |   +1.88* |  [+0.79, +2.97] | 0.001* | -0.05  | 0.861  | 1050 |
| Query-DANN (bottleneck)               |   +1.26  |  [+0.01, +2.62] | 0.060  | -0.09  | 0.834  | 1350 |
| Query InfoNCE                         |   +1.03  |  [-0.83, +2.87] | 0.265  | -3.65* | 0.001* |  400 |
| TSDAE-SAPT (task queries), final      |   -9.16* | [-10.90, -7.24] | 0.001* | -8.94* | 0.001* | 1350 |

## Notes

- LLM rewrite: gemini-3.5-flash-lite, cached rewrites of the test queries (no API calls); not
  trained. Its formal column rewrites the formal queries too, since a deployed rewriter sees every
  query.
- Query-DANN (linear): --adapter linear: a single 1024 x 1024 linear map initialised to the
  identity. Same adapter in "Query distillation score (linear)".
- Query-DANN (bottleneck): the default adapter: zero-initialised residual bottleneck 1024 -> 128 ->
  1024 on the frozen BGE-M3 query vector. Same adapter in "Query distillation score (bottleneck)".
- Query InfoNCE: --objective infonce: the full encoder fine-tuned with InfoNCE on informal queries
  against the frozen passage index, without the TSDAE reconstruction term that the method adds by
  default (weight 0.1).
- TSDAE-SAPT (task queries), final: the encoder after the last training step, NOT the best dev
  checkpoint like the other rows: dev selection keeps step 50, which has barely trained and scores
  like base. This row shows what TSDAE training itself does to retrieval.
