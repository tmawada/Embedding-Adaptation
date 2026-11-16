# Seed stability: MIRACL-id test set

- Encoder:   BGE-M3 (BAAI/bge-m3)
- Queries:   553 test queries, each in a formal and an informal version
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
| BGE-M3 base (zero-shot)               | - | 0.5362 |          - |         - |          - |            - |
| LLM rewrite                           | - | 0.5983 |      +6.21 |         - |      79.7% |            - |
| Query distillation score (bottleneck) | 5 | 0.5745 |      +3.83 |      0.26 |      49.2% |          5/0 |
| Query distillation score (linear)     | 5 | 0.5736 |      +3.75 |      0.13 |      48.1% |          5/0 |
| Query-DANN (linear)                   | 5 | 0.5553 |      +1.91 |      0.07 |      24.6% |          5/0 |
| Query-DANN (bottleneck)               | 5 | 0.5500 |      +1.38 |      0.36 |      17.7% |          3/0 |
| Query InfoNCE                         | 5 | 0.5497 |      +1.35 |      0.64 |      17.3% |          1/0 |
| TSDAE-SAPT (task queries), final      | 5 | 0.4267 |     -10.95 |      1.62 |    -140.4% |          0/5 |

## 2. Formal queries: mean over seeds and stability

Did training hurt formal queries? Change = mean difference from base in pts; Std and Sig. up/down as
in section 1.

| Model                                 | n |   Mean | Change (pts) | Std (pts) | Sig. up/down |
|---------------------------------------|--:|-------:|-------------:|----------:|-------------:|
| BGE-M3 base (zero-shot)               | - | 0.6141 |            - |         - |            - |
| LLM rewrite                           | - | 0.5926 |        -2.15 |         - |            - |
| Query distillation score (bottleneck) | 5 | 0.6145 |        +0.04 |      0.20 |          0/0 |
| Query distillation score (linear)     | 5 | 0.6148 |        +0.08 |      0.09 |          0/0 |
| Query-DANN (linear)                   | 5 | 0.6141 |        +0.00 |      0.06 |          0/0 |
| Query-DANN (bottleneck)               | 5 | 0.6121 |        -0.20 |      0.44 |          0/0 |
| Query InfoNCE                         | 5 | 0.5774 |        -3.67 |      0.63 |          0/5 |
| TSDAE-SAPT (task queries), final      | 5 | 0.5026 |       -11.15 |      1.46 |          0/5 |

## 3. Informal nDCG@10 per seed

Range = best minus worst seed, in pts.

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query distillation score (bottleneck) | 0.5710 | 0.5742 | 0.5730 | 0.5770 | 0.5771 |        0.61 |
| Query distillation score (linear)     | 0.5725 | 0.5738 | 0.5727 | 0.5758 | 0.5732 |        0.33 |
| Query-DANN (linear)                   | 0.5549 | 0.5543 | 0.5559 | 0.5553 | 0.5560 |        0.17 |
| Query-DANN (bottleneck)               | 0.5488 | 0.5504 | 0.5543 | 0.5446 | 0.5518 |        0.97 |
| Query InfoNCE                         | 0.5465 | 0.5518 | 0.5465 | 0.5598 | 0.5437 |        1.61 |
| TSDAE-SAPT (task queries), final      | 0.4446 | 0.4074 | 0.4350 | 0.4346 | 0.4119 |        3.72 |

## 4. Formal nDCG@10 per seed

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query distillation score (bottleneck) | 0.6153 | 0.6151 | 0.6111 | 0.6143 | 0.6164 |        0.53 |
| Query distillation score (linear)     | 0.6146 | 0.6138 | 0.6162 | 0.6150 | 0.6146 |        0.24 |
| Query-DANN (linear)                   | 0.6135 | 0.6151 | 0.6142 | 0.6138 | 0.6139 |        0.15 |
| Query-DANN (bottleneck)               | 0.6132 | 0.6067 | 0.6187 | 0.6110 | 0.6109 |        1.20 |
| Query InfoNCE                         | 0.5775 | 0.5786 | 0.5727 | 0.5872 | 0.5710 |        1.62 |
| TSDAE-SAPT (task queries), final      | 0.5247 | 0.4884 | 0.5092 | 0.4981 | 0.4925 |        3.63 |

## 5. All metrics, informal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| BGE-M3 base (zero-shot)               |          0.5362 |          0.6364 |          0.8986 |
| LLM rewrite                           |          0.5983 |          0.7042 |          0.9350 |
| Query distillation score (bottleneck) | 0.5745 (0.0026) | 0.6767 (0.0027) | 0.9112 (0.0019) |
| Query distillation score (linear)     | 0.5736 (0.0013) | 0.6770 (0.0009) | 0.9131 (0.0007) |
| Query-DANN (linear)                   | 0.5553 (0.0007) | 0.6604 (0.0018) | 0.9079 (0.0011) |
| Query-DANN (bottleneck)               | 0.5500 (0.0036) | 0.6560 (0.0052) | 0.8990 (0.0048) |
| Query InfoNCE                         | 0.5497 (0.0064) | 0.6521 (0.0089) | 0.8989 (0.0024) |
| TSDAE-SAPT (task queries), final      | 0.4267 (0.0162) | 0.5306 (0.0156) | 0.8004 (0.0105) |

## 6. All metrics, formal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| BGE-M3 base (zero-shot)               |          0.6141 |          0.7196 |          0.9375 |
| LLM rewrite                           |          0.5926 |          0.6987 |          0.9304 |
| Query distillation score (bottleneck) | 0.6145 (0.0020) | 0.7182 (0.0028) | 0.9405 (0.0017) |
| Query distillation score (linear)     | 0.6148 (0.0009) | 0.7201 (0.0009) | 0.9405 (0.0002) |
| Query-DANN (linear)                   | 0.6141 (0.0006) | 0.7189 (0.0006) | 0.9372 (0.0009) |
| Query-DANN (bottleneck)               | 0.6121 (0.0044) | 0.7173 (0.0053) | 0.9401 (0.0019) |
| Query InfoNCE                         | 0.5774 (0.0063) | 0.6760 (0.0059) | 0.9183 (0.0039) |
| TSDAE-SAPT (task queries), final      | 0.5026 (0.0146) | 0.6154 (0.0126) | 0.8587 (0.0121) |

## 7. Checkpoint step per seed

Optimiser step of the scored model: the best dev checkpoint unless the notes say otherwise. A step
that moves a lot between seeds means the dev selection itself is unstable.

| Model                                 |   42 |   84 |  123 |  256 |  512 |
|---------------------------------------|-----:|-----:|-----:|-----:|-----:|
| Query distillation score (bottleneck) |  950 |  950 |  750 |  950 | 1350 |
| Query distillation score (linear)     |  450 |  550 |  350 |  700 |  550 |
| Query-DANN (linear)                   | 1050 | 1350 | 1300 | 1100 | 1050 |
| Query-DANN (bottleneck)               | 1350 | 1150 |  200 |  800 |  300 |
| Query InfoNCE                         |  400 |  500 |  950 |  450 |  600 |
| TSDAE-SAPT (task queries), final      | 1350 | 1350 | 1350 | 1350 | 1350 |

## Notes

- Seeds: Only the training seed changes between runs (initialisation, batch order, sampled hard
  negatives, TSDAE noise, dropout); queries, split, caches and protocol are identical. Seed 42 is
  the run in report_test.md. Base and LLM rewrite involve no training: scored once, so they have no
  std.
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
