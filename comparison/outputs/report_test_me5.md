# Harmonized comparison: MIRACL-id test set

- Encoder:   mE5-large-instruct (intfloat/multilingual-e5-large-instruct)
- Queries:   553 test queries, each in a formal and an informal version
- Passages:  500,000, one frozen mE5-large-instruct index searched by every row
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
| mE5-large-instruct base (zero-shot)   |   0.5097 |         -  | 0.5510 |           -  |          - |
| LLM rewrite                           |   0.5437 |     +3.40* | 0.5382 |       -1.28* |      82.3% |
| Query distillation score (linear)     |   0.5281 |     +1.85* | 0.5464 |       -0.46  |      44.7% |
| Query distillation score (bottleneck) |   0.5256 |     +1.59* | 0.5485 |       -0.25  |      38.5% |
| Query-DANN (bottleneck)               |   0.5190 |     +0.94* | 0.5448 |       -0.62  |      22.6% |
| Query-DANN (linear)                   |   0.5174 |     +0.77* | 0.5493 |       -0.17  |      18.6% |
| Query InfoNCE                         |   0.4856 |     -2.41* | 0.5099 |       -4.11* |     -58.3% |
| TSDAE-SAPT (task queries), final      |   0.2186 |    -29.11* | 0.2458 |      -30.52* |    -704.7% |

## 2. All metrics, informal queries

| Model                                 | nDCG@10 | MRR@10 | Recall@100 |
|---------------------------------------|--------:|-------:|-----------:|
| mE5-large-instruct base (zero-shot)   |  0.5097 | 0.6240 |     0.8662 |
| LLM rewrite                           |  0.5437 | 0.6667 |     0.8957 |
| Query distillation score (linear)     |  0.5281 | 0.6418 |     0.8784 |
| Query distillation score (bottleneck) |  0.5256 | 0.6381 |     0.8796 |
| Query-DANN (bottleneck)               |  0.5190 | 0.6254 |     0.8740 |
| Query-DANN (linear)                   |  0.5174 | 0.6302 |     0.8717 |
| Query InfoNCE                         |  0.4856 | 0.6020 |     0.8544 |
| TSDAE-SAPT (task queries), final      |  0.2186 | 0.3053 |     0.5468 |

## 3. All metrics, formal queries

| Model                                 | nDCG@10 | MRR@10 | Recall@100 |
|---------------------------------------|--------:|-------:|-----------:|
| mE5-large-instruct base (zero-shot)   |  0.5510 | 0.6715 |     0.9045 |
| LLM rewrite                           |  0.5382 | 0.6520 |     0.8942 |
| Query distillation score (linear)     |  0.5464 | 0.6661 |     0.9060 |
| Query distillation score (bottleneck) |  0.5485 | 0.6699 |     0.9052 |
| Query-DANN (bottleneck)               |  0.5448 | 0.6606 |     0.9070 |
| Query-DANN (linear)                   |  0.5493 | 0.6702 |     0.9092 |
| Query InfoNCE                         |  0.5099 | 0.6302 |     0.8697 |
| TSDAE-SAPT (task queries), final      |  0.2458 | 0.3373 |     0.5966 |

## 4. Significance vs base and checkpoint step

Informal / Formal = nDCG@10 difference from base in pts, paired bootstrap with 1000 samples; * = p <
0.05. Step = optimiser step of the scored model: the best dev checkpoint unless the notes say
otherwise (- = not trained).

| Model                                 | Informal |           95% CI |      p |  Formal |      p | Step |
|---------------------------------------|---------:|-----------------:|-------:|--------:|-------:|-----:|
| LLM rewrite                           |   +3.40* |   [+1.89, +4.87] | 0.001* |  -1.28* | 0.001* |    - |
| Query distillation score (linear)     |   +1.85* |   [+1.05, +2.67] | 0.001* |  -0.46  | 0.072  | 1150 |
| Query distillation score (bottleneck) |   +1.59* |   [+0.70, +2.46] | 0.001* |  -0.25  | 0.276  |  800 |
| Query-DANN (bottleneck)               |   +0.94* |   [+0.01, +1.81] | 0.038* |  -0.62  | 0.114  |   50 |
| Query-DANN (linear)                   |   +0.77* |   [+0.21, +1.29] | 0.007* |  -0.17  | 0.503  |   50 |
| Query InfoNCE                         |   -2.41* |   [-4.59, -0.43] | 0.020* |  -4.11* | 0.001* |  650 |
| TSDAE-SAPT (task queries), final      |  -29.11* | [-31.85, -26.52] | 0.001* | -30.52* | 0.001* | 1350 |

## Notes

- LLM rewrite: gemini-3.5-flash-lite, cached rewrites of the test queries (no API calls); not
  trained. Its formal column rewrites the formal queries too, since a deployed rewriter sees every
  query.
- Query-DANN (bottleneck): the default adapter: zero-initialised residual bottleneck 1024 -> 128 ->
  1024 on the frozen mE5-large-instruct query vector. Same adapter in "Query distillation score
  (bottleneck)".
- Query-DANN (linear): --adapter linear: a single 1024 x 1024 linear map initialised to the
  identity. Same adapter in "Query distillation score (linear)".
- Query InfoNCE: --objective infonce: the full encoder fine-tuned with InfoNCE on informal queries
  against the frozen passage index, without the TSDAE reconstruction term that the method adds by
  default (weight 0.1).
- TSDAE-SAPT (task queries), final: the encoder after the last training step, NOT the best dev
  checkpoint like the other rows: dev selection keeps step 50, which has barely trained and scores
  like base. This row shows what TSDAE training itself does to retrieval.
- Temperature: mE5-large-instruct's cosine scores are compressed (relevant and top-ranked
  non-relevant passages both score ~0.89, random ones ~0.76), while the InfoNCE / distillation
  temperature (tau 0.05) was set for BGE-M3. The protocol is kept identical on purpose, so mE5 may
  be at a disadvantage on those methods.
