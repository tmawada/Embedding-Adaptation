# Seed stability: MIRACL-id test set

- Encoder:   mE5-large-instruct (intfloat/multilingual-e5-large-instruct)
- Queries:   553 test queries, each in a formal and an informal version
- Passages:  500,000, one frozen mE5-large-instruct index searched by every row
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
| mE5-large-instruct base (zero-shot)   | - | 0.5097 |          - |         - |          - |            - |
| LLM rewrite                           | - | 0.5437 |      +3.40 |         - |      82.3% |            - |
| Query distillation score (linear)     | 5 | 0.5263 |      +1.66 |      0.16 |      40.3% |          5/0 |
| Query distillation score (bottleneck) | 5 | 0.5243 |      +1.46 |      0.09 |      35.4% |          5/0 |
| Query-DANN (linear)                   | 5 | 0.5176 |      +0.80 |      0.08 |      19.2% |          5/0 |
| Query-DANN (bottleneck)               | 5 | 0.5166 |      +0.69 |      0.70 |      16.7% |          3/0 |
| Query InfoNCE                         | 5 | 0.4895 |      -2.02 |      0.40 |     -49.0% |          0/3 |
| TSDAE-SAPT (task queries), final      | 5 | 0.2330 |     -27.67 |      4.19 |    -669.8% |          0/5 |

## 2. Formal queries: mean over seeds and stability

Did training hurt formal queries? Change = mean difference from base in pts; Std and Sig. up/down as
in section 1.

| Model                                 | n |   Mean | Change (pts) | Std (pts) | Sig. up/down |
|---------------------------------------|--:|-------:|-------------:|----------:|-------------:|
| mE5-large-instruct base (zero-shot)   | - | 0.5510 |            - |         - |            - |
| LLM rewrite                           | - | 0.5382 |        -1.28 |         - |            - |
| Query distillation score (linear)     | 5 | 0.5470 |        -0.39 |      0.06 |          0/0 |
| Query distillation score (bottleneck) | 5 | 0.5488 |        -0.22 |      0.22 |          0/0 |
| Query-DANN (linear)                   | 5 | 0.5497 |        -0.13 |      0.07 |          0/0 |
| Query-DANN (bottleneck)               | 5 | 0.5463 |        -0.47 |      0.12 |          0/0 |
| Query InfoNCE                         | 5 | 0.5093 |        -4.17 |      0.23 |          0/5 |
| TSDAE-SAPT (task queries), final      | 5 | 0.2669 |       -28.41 |      3.66 |          0/5 |

## 3. Informal nDCG@10 per seed

Range = best minus worst seed, in pts.

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query distillation score (linear)     | 0.5281 | 0.5240 | 0.5259 | 0.5272 | 0.5263 |        0.41 |
| Query distillation score (bottleneck) | 0.5256 | 0.5247 | 0.5242 | 0.5231 | 0.5239 |        0.24 |
| Query-DANN (linear)                   | 0.5174 | 0.5184 | 0.5171 | 0.5168 | 0.5185 |        0.16 |
| Query-DANN (bottleneck)               | 0.5190 | 0.5041 | 0.5185 | 0.5209 | 0.5204 |        1.68 |
| Query InfoNCE                         | 0.4856 | 0.4936 | 0.4937 | 0.4890 | 0.4855 |        0.81 |
| TSDAE-SAPT (task queries), final      | 0.2186 | 0.2735 | 0.1800 | 0.2782 | 0.2147 |        9.81 |

## 4. Formal nDCG@10 per seed

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query distillation score (linear)     | 0.5464 | 0.5470 | 0.5467 | 0.5471 | 0.5480 |        0.16 |
| Query distillation score (bottleneck) | 0.5485 | 0.5518 | 0.5501 | 0.5465 | 0.5472 |        0.53 |
| Query-DANN (linear)                   | 0.5493 | 0.5509 | 0.5494 | 0.5494 | 0.5494 |        0.15 |
| Query-DANN (bottleneck)               | 0.5448 | 0.5454 | 0.5469 | 0.5470 | 0.5474 |        0.27 |
| Query InfoNCE                         | 0.5099 | 0.5093 | 0.5123 | 0.5058 | 0.5092 |        0.65 |
| TSDAE-SAPT (task queries), final      | 0.2458 | 0.3095 | 0.2230 | 0.2996 | 0.2564 |        8.65 |

## 5. All metrics, informal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| mE5-large-instruct base (zero-shot)   |          0.5097 |          0.6240 |          0.8662 |
| LLM rewrite                           |          0.5437 |          0.6667 |          0.8957 |
| Query distillation score (linear)     | 0.5263 (0.0016) | 0.6396 (0.0023) | 0.8790 (0.0021) |
| Query distillation score (bottleneck) | 0.5243 (0.0009) | 0.6381 (0.0017) | 0.8801 (0.0018) |
| Query-DANN (linear)                   | 0.5176 (0.0008) | 0.6293 (0.0014) | 0.8724 (0.0005) |
| Query-DANN (bottleneck)               | 0.5166 (0.0070) | 0.6247 (0.0031) | 0.8722 (0.0067) |
| Query InfoNCE                         | 0.4895 (0.0040) | 0.6034 (0.0043) | 0.8530 (0.0039) |
| TSDAE-SAPT (task queries), final      | 0.2330 (0.0419) | 0.3245 (0.0584) | 0.5733 (0.0567) |

## 6. All metrics, formal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| mE5-large-instruct base (zero-shot)   |          0.5510 |          0.6715 |          0.9045 |
| LLM rewrite                           |          0.5382 |          0.6520 |          0.8942 |
| Query distillation score (linear)     | 0.5470 (0.0006) | 0.6663 (0.0009) | 0.9062 (0.0007) |
| Query distillation score (bottleneck) | 0.5488 (0.0022) | 0.6695 (0.0024) | 0.9044 (0.0005) |
| Query-DANN (linear)                   | 0.5497 (0.0007) | 0.6711 (0.0010) | 0.9075 (0.0011) |
| Query-DANN (bottleneck)               | 0.5463 (0.0012) | 0.6633 (0.0020) | 0.9044 (0.0043) |
| Query InfoNCE                         | 0.5093 (0.0023) | 0.6291 (0.0052) | 0.8688 (0.0074) |
| TSDAE-SAPT (task queries), final      | 0.2669 (0.0366) | 0.3644 (0.0461) | 0.6265 (0.0444) |

## 7. Checkpoint step per seed

Optimiser step of the scored model: the best dev checkpoint unless the notes say otherwise. A step
that moves a lot between seeds means the dev selection itself is unstable.

| Model                                 |   42 |   84 |  123 |  256 |  512 |
|---------------------------------------|-----:|-----:|-----:|-----:|-----:|
| Query distillation score (linear)     | 1150 |  600 |  750 |  950 |  500 |
| Query distillation score (bottleneck) |  800 |  800 |  700 |  300 |  500 |
| Query-DANN (linear)                   |   50 |   50 |   50 |   50 |   50 |
| Query-DANN (bottleneck)               |   50 | 1100 |   50 |   50 |   50 |
| Query InfoNCE                         |  650 |  200 |  800 |  450 |  350 |
| TSDAE-SAPT (task queries), final      | 1350 | 1350 | 1350 | 1350 | 1350 |

## Notes

- Seeds: Only the training seed changes between runs (initialisation, batch order, sampled hard
  negatives, TSDAE noise, dropout); queries, split, caches and protocol are identical. Seed 42 is
  the run in report_test_me5.md. Base and LLM rewrite involve no training: scored once, so they have
  no std.
- LLM rewrite: gemini-3.5-flash-lite, cached rewrites of the test queries (no API calls); not
  trained. Its formal column rewrites the formal queries too, since a deployed rewriter sees every
  query.
- Query-DANN (linear): --adapter linear: a single 1024 x 1024 linear map initialised to the
  identity. Same adapter in "Query distillation score (linear)".
- Query-DANN (bottleneck): the default adapter: zero-initialised residual bottleneck 1024 -> 128 ->
  1024 on the frozen mE5-large-instruct query vector. Same adapter in "Query distillation score
  (bottleneck)".
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
