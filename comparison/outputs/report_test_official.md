# Harmonized comparison: MIRACL-id test set

- Encoder:   BGE-M3 (BAAI/bge-m3)
- Queries:   867 test queries, each in a formal and an informal version
- Dataset:   data/official: trained on official MIRACL-id train, checkpoint on our dev split, tested
             on official MIRACL-id dev queries (never used to train the encoders)
- Informal:  template style (the style of every training query, mostly '... sih')
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
| BGE-M3 base (zero-shot)               |   0.5626 |         -  | 0.6053 |           -  |          - |
| LLM rewrite                           |   0.6007 |     +3.80* | 0.5963 |       -0.90* |      89.2% |
| Query distillation score (linear)     |   0.5802 |     +1.75* | 0.6010 |       -0.42* |      41.1% |
| Query distillation score (bottleneck) |   0.5794 |     +1.68* | 0.6005 |       -0.47* |      39.3% |
| Query InfoNCE                         |   0.5792 |     +1.65* | 0.5902 |       -1.50* |      38.7% |
| Query-DANN (linear)                   |   0.5715 |     +0.89* | 0.6038 |       -0.14  |      20.9% |
| Query-DANN (bottleneck)               |   0.5685 |     +0.58  | 0.5992 |       -0.61  |      13.7% |
| TSDAE-SAPT (task queries), final      |   0.4670 |     -9.56* | 0.5229 |       -8.23* |    -224.3% |

## 2. All metrics, informal queries

| Model                                 | nDCG@10 | MRR@10 | Recall@100 |
|---------------------------------------|--------:|-------:|-----------:|
| BGE-M3 base (zero-shot)               |  0.5626 | 0.6719 |     0.8870 |
| LLM rewrite                           |  0.6007 | 0.6995 |     0.9154 |
| Query distillation score (linear)     |  0.5802 | 0.6812 |     0.9043 |
| Query distillation score (bottleneck) |  0.5794 | 0.6812 |     0.9040 |
| Query InfoNCE                         |  0.5792 | 0.6783 |     0.8970 |
| Query-DANN (linear)                   |  0.5715 | 0.6760 |     0.8989 |
| Query-DANN (bottleneck)               |  0.5685 | 0.6735 |     0.8975 |
| TSDAE-SAPT (task queries), final      |  0.4670 | 0.5735 |     0.8226 |

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
| LLM rewrite                           |   +3.80* |  [+2.83, +4.88] | 0.001* | -0.90* | 0.008* |    - |
| Query distillation score (linear)     |   +1.75* |  [+1.13, +2.35] | 0.001* | -0.42* | 0.037* |  550 |
| Query distillation score (bottleneck) |   +1.68* |  [+1.01, +2.29] | 0.001* | -0.47* | 0.010* | 1250 |
| Query InfoNCE                         |   +1.65* |  [+0.31, +3.05] | 0.022* | -1.50* | 0.024* |  900 |
| Query-DANN (linear)                   |   +0.89* |  [+0.13, +1.60] | 0.017* | -0.14  | 0.610  |  800 |
| Query-DANN (bottleneck)               |   +0.58  |  [-0.35, +1.54] | 0.221  | -0.61  | 0.079  | 1350 |
| TSDAE-SAPT (task queries), final      |   -9.56* | [-10.95, -8.17] | 0.001* | -8.23* | 0.001* | 1350 |

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
