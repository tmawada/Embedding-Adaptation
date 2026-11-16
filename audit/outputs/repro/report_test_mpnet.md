# Harmonized comparison: MIRACL-id test set

- Encoder:   multilingual-mpnet (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- Queries:   553 test queries, each in a formal and an informal version
- Passages:  500,000, one frozen multilingual-mpnet index searched by every row
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
| multilingual-mpnet base (zero-shot)   |   0.1948 |         -  | 0.2935 |           -  |          - |
| Query InfoNCE                         |   0.3750 |    +18.02* | 0.4012 |      +10.77* |     182.6% |
| LLM rewrite                           |   0.3015 |    +10.67* | 0.3066 |       +1.31* |     108.1% |
| Query-DANN (linear)                   |   0.2875 |     +9.26* | 0.3723 |       +7.88* |      93.9% |
| Query-DANN (bottleneck)               |   0.2796 |     +8.47* | 0.3454 |       +5.19* |      85.9% |
| Query distillation score (linear)     |   0.2266 |     +3.18* | 0.2968 |       +0.33  |      32.2% |
| Query distillation score (bottleneck) |   0.2264 |     +3.15* | 0.2869 |       -0.66  |      32.0% |
| TSDAE-SAPT (task queries), final      |   0.1506 |     -4.43* | 0.2410 |       -5.25* |     -44.8% |

## 2. All metrics, informal queries

| Model                                 | nDCG@10 | MRR@10 | Recall@100 |
|---------------------------------------|--------:|-------:|-----------:|
| multilingual-mpnet base (zero-shot)   |  0.1948 | 0.2483 |     0.5001 |
| Query InfoNCE                         |  0.3750 | 0.4744 |     0.7151 |
| LLM rewrite                           |  0.3015 | 0.3755 |     0.6548 |
| Query-DANN (linear)                   |  0.2875 | 0.3702 |     0.6376 |
| Query-DANN (bottleneck)               |  0.2796 | 0.3575 |     0.6389 |
| Query distillation score (linear)     |  0.2266 | 0.2939 |     0.5728 |
| Query distillation score (bottleneck) |  0.2264 | 0.2928 |     0.5735 |
| TSDAE-SAPT (task queries), final      |  0.1506 | 0.1973 |     0.4685 |

## 3. All metrics, formal queries

| Model                                 | nDCG@10 | MRR@10 | Recall@100 |
|---------------------------------------|--------:|-------:|-----------:|
| multilingual-mpnet base (zero-shot)   |  0.2935 | 0.3711 |     0.6561 |
| Query InfoNCE                         |  0.4012 | 0.5048 |     0.7644 |
| LLM rewrite                           |  0.3066 | 0.3822 |     0.6701 |
| Query-DANN (linear)                   |  0.3723 | 0.4754 |     0.7372 |
| Query-DANN (bottleneck)               |  0.3454 | 0.4322 |     0.7333 |
| Query distillation score (linear)     |  0.2968 | 0.3822 |     0.6627 |
| Query distillation score (bottleneck) |  0.2869 | 0.3674 |     0.6572 |
| TSDAE-SAPT (task queries), final      |  0.2410 | 0.3130 |     0.5756 |

## 4. Significance vs base and checkpoint step

nDCG@10 difference from base in pts, paired bootstrap with 1000 samples; * = p < 0.05. Step =
optimiser step of the scored model: the best dev checkpoint unless the notes say otherwise (- = not
trained).

| Model                                 | Informal gain |           95% CI |      p | Formal change |      p | Step |
|---------------------------------------|--------------:|-----------------:|-------:|--------------:|-------:|-----:|
| Query InfoNCE                         |       +18.02* | [+15.40, +20.57] | 0.001* |       +10.77* | 0.001* | 1050 |
| LLM rewrite                           |       +10.67* |  [+8.66, +12.61] | 0.001* |        +1.31* | 0.027* |    - |
| Query-DANN (linear)                   |        +9.26* |  [+7.60, +10.86] | 0.001* |        +7.88* | 0.001* |  900 |
| Query-DANN (bottleneck)               |        +8.47* |  [+6.74, +10.07] | 0.001* |        +5.19* | 0.001* |  600 |
| Query distillation score (linear)     |        +3.18* |   [+2.31, +4.16] | 0.001* |        +0.33  | 0.369  | 1100 |
| Query distillation score (bottleneck) |        +3.15* |   [+2.19, +4.32] | 0.001* |        -0.66  | 0.073  |  450 |
| TSDAE-SAPT (task queries), final      |        -4.43* |   [-6.15, -2.64] | 0.001* |        -5.25* | 0.001* | 1350 |

## Notes

- Query InfoNCE: --objective infonce: the full encoder fine-tuned with InfoNCE on informal queries
  against the frozen passage index, without the TSDAE reconstruction term that the method adds by
  default (weight 0.1).
- LLM rewrite: gemini-3.5-flash-lite, cached rewrites of the test queries (no API calls); not
  trained. Its formal column rewrites the formal queries too, since a deployed rewriter sees every
  query.
- Query-DANN (linear): --adapter linear: a single 768 x 768 linear map initialised to the identity.
  Same adapter in "Query distillation score (linear)".
- Query-DANN (bottleneck): the default adapter: zero-initialised residual bottleneck 768 -> 128 ->
  768 on the frozen multilingual-mpnet query vector. Same adapter in "Query distillation score
  (bottleneck)".
- TSDAE-SAPT (task queries), final: the encoder after the last training step, NOT the best dev
  checkpoint like the other rows: dev selection keeps step 50, which has barely trained and scores
  like base. This row shows what TSDAE training itself does to retrieval.
