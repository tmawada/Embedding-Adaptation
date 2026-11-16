# Seed stability: MIRACL-id test set

- Encoder:   mE5-large-instruct (intfloat/multilingual-e5-large-instruct)
- Queries:   867 test queries, each in a formal and an informal version
- Dataset:   data/official: trained on official MIRACL-id train, checkpoint on our dev split, tested
             on official MIRACL-id dev queries (never used to train the encoders)
- Informal:  varied style (sampled per query: slang, abbreviations, typos, word order)
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
| mE5-large-instruct base (zero-shot)   | - | 0.5267 |          - |         - |          - |            - |
| LLM rewrite                           | - | 0.5634 |      +3.67 |         - |      85.9% |            - |
| Query distillation score (bottleneck) | 5 | 0.5388 |      +1.20 |      0.18 |      28.2% |          5/0 |
| Query distillation score (linear)     | 5 | 0.5359 |      +0.92 |      0.12 |      21.5% |          5/0 |
| Query-DANN (bottleneck)               | 5 | 0.5348 |      +0.81 |      0.15 |      18.8% |          4/0 |
| Query-DANN (linear)                   | 5 | 0.5338 |      +0.70 |      0.10 |      16.5% |          5/0 |
| Query InfoNCE                         | 5 | 0.5104 |      -1.63 |      0.49 |     -38.3% |          0/3 |
| TSDAE-SAPT (task queries), final      | 5 | 0.2249 |     -30.19 |      1.93 |    -706.7% |          0/5 |

## 2. Formal queries: mean over seeds and stability

Did training hurt formal queries? Change = mean difference from base in pts; Std and Sig. up/down as
in section 1.

| Model                                 | n |   Mean | Change (pts) | Std (pts) | Sig. up/down |
|---------------------------------------|--:|-------:|-------------:|----------:|-------------:|
| mE5-large-instruct base (zero-shot)   | - | 0.5694 |            - |         - |            - |
| LLM rewrite                           | - | 0.5642 |        -0.52 |         - |            - |
| Query distillation score (bottleneck) | 5 | 0.5687 |        -0.07 |      0.13 |          0/0 |
| Query distillation score (linear)     | 5 | 0.5671 |        -0.24 |      0.06 |          0/0 |
| Query-DANN (bottleneck)               | 5 | 0.5692 |        -0.03 |      0.09 |          0/0 |
| Query-DANN (linear)                   | 5 | 0.5714 |        +0.19 |      0.07 |          0/0 |
| Query InfoNCE                         | 5 | 0.5377 |        -3.18 |      0.12 |          0/5 |
| TSDAE-SAPT (task queries), final      | 5 | 0.2813 |       -28.81 |      3.42 |          0/5 |

## 3. Informal nDCG@10 per seed

Range = best minus worst seed, in pts.

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query distillation score (bottleneck) | 0.5395 | 0.5368 | 0.5402 | 0.5369 | 0.5405 |        0.37 |
| Query distillation score (linear)     | 0.5346 | 0.5364 | 0.5372 | 0.5366 | 0.5346 |        0.26 |
| Query-DANN (bottleneck)               | 0.5331 | 0.5366 | 0.5336 | 0.5360 | 0.5347 |        0.36 |
| Query-DANN (linear)                   | 0.5322 | 0.5349 | 0.5341 | 0.5333 | 0.5344 |        0.27 |
| Query InfoNCE                         | 0.5138 | 0.5173 | 0.5061 | 0.5083 | 0.5064 |        1.12 |
| TSDAE-SAPT (task queries), final      | 0.2327 | 0.2176 | 0.2371 | 0.2424 | 0.1946 |        4.78 |

## 4. Formal nDCG@10 per seed

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query distillation score (bottleneck) | 0.5685 | 0.5674 | 0.5696 | 0.5676 | 0.5706 |        0.32 |
| Query distillation score (linear)     | 0.5660 | 0.5671 | 0.5673 | 0.5675 | 0.5674 |        0.15 |
| Query-DANN (bottleneck)               | 0.5679 | 0.5687 | 0.5697 | 0.5701 | 0.5695 |        0.22 |
| Query-DANN (linear)                   | 0.5716 | 0.5724 | 0.5711 | 0.5707 | 0.5711 |        0.17 |
| Query InfoNCE                         | 0.5368 | 0.5383 | 0.5362 | 0.5380 | 0.5391 |        0.29 |
| TSDAE-SAPT (task queries), final      | 0.3129 | 0.2617 | 0.3020 | 0.2990 | 0.2309 |        8.21 |

## 5. All metrics, informal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| mE5-large-instruct base (zero-shot)   |          0.5267 |          0.6346 |          0.8726 |
| LLM rewrite                           |          0.5634 |          0.6734 |          0.9058 |
| Query distillation score (bottleneck) | 0.5388 (0.0018) | 0.6483 (0.0041) | 0.8823 (0.0022) |
| Query distillation score (linear)     | 0.5359 (0.0012) | 0.6457 (0.0016) | 0.8813 (0.0009) |
| Query-DANN (bottleneck)               | 0.5348 (0.0015) | 0.6453 (0.0018) | 0.8779 (0.0004) |
| Query-DANN (linear)                   | 0.5338 (0.0010) | 0.6435 (0.0013) | 0.8802 (0.0004) |
| Query InfoNCE                         | 0.5104 (0.0049) | 0.6151 (0.0059) | 0.8530 (0.0044) |
| TSDAE-SAPT (task queries), final      | 0.2249 (0.0193) | 0.3031 (0.0241) | 0.5559 (0.0295) |

## 6. All metrics, formal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| mE5-large-instruct base (zero-shot)   |          0.5694 |          0.6751 |          0.9070 |
| LLM rewrite                           |          0.5642 |          0.6761 |          0.9053 |
| Query distillation score (bottleneck) | 0.5687 (0.0013) | 0.6745 (0.0023) | 0.9092 (0.0021) |
| Query distillation score (linear)     | 0.5671 (0.0006) | 0.6724 (0.0004) | 0.9096 (0.0010) |
| Query-DANN (bottleneck)               | 0.5692 (0.0009) | 0.6746 (0.0019) | 0.9115 (0.0007) |
| Query-DANN (linear)                   | 0.5714 (0.0007) | 0.6783 (0.0008) | 0.9095 (0.0007) |
| Query InfoNCE                         | 0.5377 (0.0012) | 0.6436 (0.0039) | 0.8766 (0.0055) |
| TSDAE-SAPT (task queries), final      | 0.2813 (0.0342) | 0.3745 (0.0469) | 0.6158 (0.0329) |

## 7. Checkpoint step per seed

Optimiser step of the scored model: the best dev checkpoint unless the notes say otherwise. A step
that moves a lot between seeds means the dev selection itself is unstable.

| Model                                 |   42 |   84 |  123 |  256 |  512 |
|---------------------------------------|-----:|-----:|-----:|-----:|-----:|
| Query distillation score (bottleneck) |  800 |  900 | 1050 |  600 |  750 |
| Query distillation score (linear)     |  350 | 1050 |  600 |  850 |  400 |
| Query-DANN (bottleneck)               |   50 |   50 |   50 |   50 |   50 |
| Query-DANN (linear)                   |   50 |   50 |   50 |   50 |   50 |
| Query InfoNCE                         |  950 |  650 |  700 | 1200 |  250 |
| TSDAE-SAPT (task queries), final      | 1350 | 1350 | 1350 | 1350 | 1350 |

## Notes

- Seeds: Only the training seed changes between runs (initialisation, batch order, sampled hard
  negatives, TSDAE noise, dropout); queries, split, caches and protocol are identical. Seed 42 is
  the run in report_test_me5_official_varied.md. Base and LLM rewrite involve no training: scored
  once, so they have no std.
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
