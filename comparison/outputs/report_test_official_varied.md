# Harmonized comparison: MIRACL-id test set

- Encoder:   BGE-M3 (BAAI/bge-m3)
- Queries:   867 test queries, each in a formal and an informal version
- Dataset:   data/official: trained on official MIRACL-id train, checkpoint on our dev split, tested
             on official MIRACL-id dev queries (never used to train the encoders)
- Informal:  varied style (sampled per query: slang, abbreviations, typos, word order)
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
| BGE-M3 base (zero-shot)               |   0.5264 |         -  | 0.6053 |           -  |          - |
| LLM rewrite                           |   0.5962 |     +6.98* | 0.5963 |       -0.90* |      88.5% |
| Query distillation score (linear)     |   0.5537 |     +2.73* | 0.6010 |       -0.42* |      34.6% |
| Query distillation score (bottleneck) |   0.5513 |     +2.49* | 0.6005 |       -0.47* |      31.5% |
| Query InfoNCE                         |   0.5479 |     +2.15* | 0.5902 |       -1.50* |      27.2% |
| Query-DANN (linear)                   |   0.5393 |     +1.29* | 0.6038 |       -0.14  |      16.3% |
| Query-DANN (bottleneck)               |   0.5343 |     +0.79  | 0.5992 |       -0.61  |      10.0% |
| TSDAE-SAPT (task queries), final      |   0.4264 |    -10.00* | 0.5229 |       -8.23* |    -126.8% |

## 2. All metrics, informal queries

| Model                                 | nDCG@10 | MRR@10 | Recall@100 |
|---------------------------------------|--------:|-------:|-----------:|
| BGE-M3 base (zero-shot)               |  0.5264 | 0.6247 |     0.8517 |
| LLM rewrite                           |  0.5962 | 0.6936 |     0.9113 |
| Query distillation score (linear)     |  0.5537 | 0.6548 |     0.8739 |
| Query distillation score (bottleneck) |  0.5513 | 0.6522 |     0.8679 |
| Query InfoNCE                         |  0.5479 | 0.6404 |     0.8805 |
| Query-DANN (linear)                   |  0.5393 | 0.6361 |     0.8674 |
| Query-DANN (bottleneck)               |  0.5343 | 0.6331 |     0.8572 |
| TSDAE-SAPT (task queries), final      |  0.4264 | 0.5241 |     0.7818 |

## 3. All metrics, formal queries

| Model                                 | nDCG@10 | MRR@10 | Recall@100 |
|---------------------------------------|--------:|-------:|-----------:|
| BGE-M3 base (zero-shot)               |  0.6053 | 0.7048 |     0.9129 |
| LLM rewrite                           |  0.5963 | 0.6965 |     0.9138 |
| Query distillation score (linear)     |  0.6010 | 0.7001 |     0.9111 |
| Query distillation score (bottleneck) |  0.6005 | 0.7004 |     0.9099 |
| Query InfoNCE                         |  0.5902 | 0.6926 |     0.9041 |
| Query-DANN (linear)                   |  0.6038 | 0.7002 |     0.9185 |
| Query-DANN (bottleneck)               |  0.5992 | 0.6950 |     0.9160 |
| TSDAE-SAPT (task queries), final      |  0.5229 | 0.6243 |     0.8679 |

## 4. Significance vs base and checkpoint step

Informal / Formal = nDCG@10 difference from base in pts, paired bootstrap with 1000 samples; * = p <
0.05. Step = optimiser step of the scored model: the best dev checkpoint unless the notes say
otherwise (- = not trained).

| Model                                 | Informal |          95% CI |      p | Formal |      p | Step |
|---------------------------------------|---------:|----------------:|-------:|-------:|-------:|-----:|
| LLM rewrite                           |   +6.98* |  [+5.61, +8.29] | 0.001* | -0.90* | 0.008* |    - |
| Query distillation score (linear)     |   +2.73* |  [+2.07, +3.34] | 0.001* | -0.42* | 0.037* |  550 |
| Query distillation score (bottleneck) |   +2.49* |  [+1.82, +3.21] | 0.001* | -0.47* | 0.010* | 1250 |
| Query InfoNCE                         |   +2.15* |  [+0.59, +3.59] | 0.005* | -1.50* | 0.024* |  900 |
| Query-DANN (linear)                   |   +1.29* |  [+0.53, +2.06] | 0.003* | -0.14  | 0.610  |  800 |
| Query-DANN (bottleneck)               |   +0.79  |  [-0.26, +1.79] | 0.119  | -0.61  | 0.079  | 1350 |
| TSDAE-SAPT (task queries), final      |  -10.00* | [-11.49, -8.52] | 0.001* | -8.23* | 0.001* | 1350 |

## Notes

- LLM rewrite: gemini-3.5-flash-lite, cached rewrites of the test queries (no API calls); not
  trained. Its formal column rewrites the formal queries too, since a deployed rewriter sees every
  query.
- Query InfoNCE: --objective infonce: the full encoder fine-tuned with InfoNCE on informal queries
  against the frozen passage index, without the TSDAE reconstruction term that the method adds by
  default (weight 0.1).
- Query-DANN (linear): --adapter linear: a single 1024 x 1024 linear map initialised to the
  identity. Same adapter in "Query distillation score (linear)".
- Query-DANN (bottleneck): the default adapter: zero-initialised residual bottleneck 1024 -> 128 ->
  1024 on the frozen BGE-M3 query vector. Same adapter in "Query distillation score (bottleneck)".
- TSDAE-SAPT (task queries), final: the encoder after the last training step, NOT the best dev
  checkpoint like the other rows: dev selection keeps step 50, which has barely trained and scores
  like base. This row shows what TSDAE training itself does to retrieval.
