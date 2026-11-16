# Seed stability: MIRACL-id test set

- Encoder:   BGE-M3 (BAAI/bge-m3)
- Queries:   867 test queries, each in a formal and an informal version
- Dataset:   data/official: trained on official MIRACL-id train, checkpoint on our dev split, tested
             on official MIRACL-id dev queries (never used to train the encoders)
- Informal:  varied style (sampled per query: slang, abbreviations, typos, word order)
- Passages:  500,000, one frozen BGE-M3 index searched by every row
- Seeds:     42, 84, 123, 256, 512; every trained model is trained once per seed
- Scores:    nDCG@10 unless stated otherwise; pts = nDCG@10 x 100

## Training protocol (every trained model)

- Steps:       1350 optimiser steps, effective batch 64
- Optimiser:   AdamW, lr 3e-05, weight decay 0.01, grad clip 1.0
- Schedule:    cosine down to 1e-06 after 10% linear warmup
- Precision:   bf16, max_length 128 (encoder fine-tuning)
- Seeds:       42, 84, 123, 256, 512 (one training run per seed and model)
- Checkpoint:  best dev informal nDCG@10, checked every 50 steps

## 1. Informal queries: mean over seeds and stability

Sorted by mean informal nDCG@10. n = seeds scored. Gain = mean difference from base in pts. Gap
closed = mean share of the base model's formal-informal gap recovered. Std = sample standard
deviation across seeds (n - 1), in pts; it is the same for the score and for the gain, because the
base row does not depend on the seed. Sig. up/down = how many seeds are significantly better / worse
than base on their own (paired bootstrap on nDCG@10, p < 0.05). A model is stable when its std is
small next to its gain and all seeds agree in Sig. up/down.

| Model                                 | n |   Mean | Gain (pts) | Std (pts) | Gap closed | Sig. up/down |
|---------------------------------------|--:|-------:|-----------:|----------:|-----------:|-------------:|
| BGE-M3 base (zero-shot)               | - | 0.5264 |          - |         - |          - |            - |
| LLM rewrite                           | - | 0.5962 |      +6.98 |         - |      88.5% |            - |
| Query distillation score (linear)     | 5 | 0.5538 |      +2.74 |      0.07 |      34.8% |          5/0 |
| Query distillation score (bottleneck) | 5 | 0.5524 |      +2.60 |      0.12 |      32.9% |          5/0 |
| Query InfoNCE                         | 5 | 0.5508 |      +2.43 |      0.25 |      30.9% |          5/0 |
| Query-DANN (linear)                   | 5 | 0.5404 |      +1.40 |      0.16 |      17.7% |          5/0 |
| Query-DANN (bottleneck)               | 5 | 0.5338 |      +0.74 |      0.34 |       9.4% |          2/0 |
| TSDAE-SAPT (task queries), final      | 5 | 0.3849 |     -14.15 |      2.47 |    -179.5% |          0/5 |

## 2. Formal queries: mean over seeds and stability

Did training hurt formal queries? Change = mean difference from base in pts; Std and Sig. up/down as
in section 1.

| Model                                 | n |   Mean | Change (pts) | Std (pts) | Sig. up/down |
|---------------------------------------|--:|-------:|-------------:|----------:|-------------:|
| BGE-M3 base (zero-shot)               | - | 0.6053 |            - |         - |            - |
| LLM rewrite                           | - | 0.5963 |        -0.90 |         - |            - |
| Query distillation score (linear)     | 5 | 0.6015 |        -0.38 |      0.06 |          0/2 |
| Query distillation score (bottleneck) | 5 | 0.6017 |        -0.36 |      0.08 |          0/1 |
| Query InfoNCE                         | 5 | 0.5882 |        -1.71 |      0.36 |          0/5 |
| Query-DANN (linear)                   | 5 | 0.6055 |        +0.03 |      0.10 |          0/0 |
| Query-DANN (bottleneck)               | 5 | 0.6041 |        -0.12 |      0.29 |          0/0 |
| TSDAE-SAPT (task queries), final      | 5 | 0.5011 |       -10.42 |      1.41 |          0/5 |

## 3. Informal nDCG@10 per seed

Range = best minus worst seed, in pts.

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query distillation score (linear)     | 0.5537 | 0.5547 | 0.5543 | 0.5532 | 0.5533 |        0.16 |
| Query distillation score (bottleneck) | 0.5513 | 0.5516 | 0.5530 | 0.5518 | 0.5543 |        0.30 |
| Query InfoNCE                         | 0.5479 | 0.5513 | 0.5514 | 0.5544 | 0.5488 |        0.65 |
| Query-DANN (linear)                   | 0.5393 | 0.5394 | 0.5432 | 0.5402 | 0.5398 |        0.39 |
| Query-DANN (bottleneck)               | 0.5343 | 0.5334 | 0.5378 | 0.5351 | 0.5286 |        0.92 |
| TSDAE-SAPT (task queries), final      | 0.4264 | 0.3620 | 0.3784 | 0.3731 | 0.3845 |        6.44 |

## 4. Formal nDCG@10 per seed

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query distillation score (linear)     | 0.6010 | 0.6017 | 0.6006 | 0.6019 | 0.6020 |        0.13 |
| Query distillation score (bottleneck) | 0.6005 | 0.6014 | 0.6017 | 0.6019 | 0.6026 |        0.21 |
| Query InfoNCE                         | 0.5902 | 0.5916 | 0.5835 | 0.5904 | 0.5854 |        0.81 |
| Query-DANN (linear)                   | 0.6038 | 0.6058 | 0.6060 | 0.6056 | 0.6065 |        0.27 |
| Query-DANN (bottleneck)               | 0.5992 | 0.6045 | 0.6060 | 0.6067 | 0.6042 |        0.75 |
| TSDAE-SAPT (task queries), final      | 0.5229 | 0.4840 | 0.5015 | 0.5008 | 0.4960 |        3.90 |

## 5. All metrics, informal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| BGE-M3 base (zero-shot)               |          0.5264 |          0.6247 |          0.8517 |
| LLM rewrite                           |          0.5962 |          0.6936 |          0.9113 |
| Query distillation score (linear)     | 0.5538 (0.0007) | 0.6549 (0.0012) | 0.8732 (0.0005) |
| Query distillation score (bottleneck) | 0.5524 (0.0012) | 0.6553 (0.0024) | 0.8714 (0.0024) |
| Query InfoNCE                         | 0.5508 (0.0025) | 0.6430 (0.0038) | 0.8819 (0.0011) |
| Query-DANN (linear)                   | 0.5404 (0.0016) | 0.6365 (0.0018) | 0.8683 (0.0014) |
| Query-DANN (bottleneck)               | 0.5338 (0.0034) | 0.6314 (0.0020) | 0.8625 (0.0033) |
| TSDAE-SAPT (task queries), final      | 0.3849 (0.0247) | 0.4807 (0.0263) | 0.7444 (0.0225) |

## 6. All metrics, formal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| BGE-M3 base (zero-shot)               |          0.6053 |          0.7048 |          0.9129 |
| LLM rewrite                           |          0.5963 |          0.6965 |          0.9138 |
| Query distillation score (linear)     | 0.6015 (0.0006) | 0.7007 (0.0006) | 0.9115 (0.0007) |
| Query distillation score (bottleneck) | 0.6017 (0.0008) | 0.7018 (0.0010) | 0.9116 (0.0018) |
| Query InfoNCE                         | 0.5882 (0.0036) | 0.6878 (0.0045) | 0.9073 (0.0023) |
| Query-DANN (linear)                   | 0.6055 (0.0010) | 0.7034 (0.0019) | 0.9198 (0.0008) |
| Query-DANN (bottleneck)               | 0.6041 (0.0029) | 0.7024 (0.0045) | 0.9157 (0.0026) |
| TSDAE-SAPT (task queries), final      | 0.5011 (0.0141) | 0.6072 (0.0120) | 0.8451 (0.0141) |

## 7. Checkpoint step per seed

Optimiser step of the scored model: the best dev checkpoint unless the notes say otherwise. A step
that moves a lot between seeds means the dev selection itself is unstable.

| Model                                 |   42 |   84 |  123 |  256 |  512 |
|---------------------------------------|-----:|-----:|-----:|-----:|-----:|
| Query distillation score (linear)     |  550 |  650 |  650 |  550 |  500 |
| Query distillation score (bottleneck) | 1250 |  700 |  900 |  750 | 1000 |
| Query InfoNCE                         |  900 | 1000 | 1250 | 1000 |  750 |
| Query-DANN (linear)                   |  800 | 1000 | 1250 | 1350 | 1250 |
| Query-DANN (bottleneck)               | 1350 |  300 |  250 |  300 |  950 |
| TSDAE-SAPT (task queries), final      | 1350 | 1350 | 1350 | 1350 | 1350 |

## Notes

- Seeds: Only the training seed changes between runs (initialisation, batch order, sampled hard
  negatives, TSDAE noise, dropout); queries, split, caches and protocol are identical. Seed 42 is
  the run in report_test_official_varied.md. Base and LLM rewrite involve no training: scored once,
  so they have no std.
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
