# Seed stability: MIRACL-id test set

- Encoder:   multilingual-mpnet (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- Queries:   867 test queries, each in a formal and an informal version
- Dataset:   data/official: trained on official MIRACL-id train, checkpoint on our dev split, tested
             on official MIRACL-id dev queries (never used to train the encoders)
- Informal:  varied style (sampled per query: slang, abbreviations, typos, word order)
- Passages:  500,000, one frozen multilingual-mpnet index searched by every row
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
| multilingual-mpnet base (zero-shot)   | - | 0.1879 |          - |         - |          - |            - |
| Query InfoNCE                         | 5 | 0.3553 |     +16.74 |      0.22 |     138.0% |          5/0 |
| LLM rewrite                           | - | 0.3096 |     +12.17 |         - |     100.3% |            - |
| Query-DANN (linear)                   | 5 | 0.2806 |      +9.27 |      0.08 |      76.4% |          5/0 |
| Query-DANN (bottleneck)               | 5 | 0.2745 |      +8.66 |      0.31 |      71.4% |          5/0 |
| Query distillation score (bottleneck) | 5 | 0.2365 |      +4.86 |      0.19 |      40.0% |          5/0 |
| Query distillation score (linear)     | 5 | 0.2363 |      +4.84 |      0.03 |      39.9% |          5/0 |
| TSDAE-SAPT (task queries), final      | 5 | 0.1485 |      -3.94 |      0.29 |     -32.5% |          0/5 |

## 2. Formal queries: mean over seeds and stability

Did training hurt formal queries? Change = mean difference from base in pts; Std and Sig. up/down as
in section 1.

| Model                                 | n |   Mean | Change (pts) | Std (pts) | Sig. up/down |
|---------------------------------------|--:|-------:|-------------:|----------:|-------------:|
| multilingual-mpnet base (zero-shot)   | - | 0.3092 |            - |         - |            - |
| Query InfoNCE                         | 5 | 0.4125 |       +10.33 |      0.21 |          5/0 |
| LLM rewrite                           | - | 0.3133 |        +0.41 |         - |            - |
| Query-DANN (linear)                   | 5 | 0.3949 |        +8.57 |      0.05 |          5/0 |
| Query-DANN (bottleneck)               | 5 | 0.3632 |        +5.40 |      0.51 |          5/0 |
| Query distillation score (bottleneck) | 5 | 0.3129 |        +0.37 |      0.29 |          1/0 |
| Query distillation score (linear)     | 5 | 0.3208 |        +1.16 |      0.02 |          5/0 |
| TSDAE-SAPT (task queries), final      | 5 | 0.2467 |        -6.26 |      0.74 |          0/5 |

## 3. Informal nDCG@10 per seed

Range = best minus worst seed, in pts.

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query InfoNCE                         | 0.3570 | 0.3551 | 0.3563 | 0.3564 | 0.3514 |        0.56 |
| Query-DANN (linear)                   | 0.2806 | 0.2806 | 0.2818 | 0.2797 | 0.2801 |        0.21 |
| Query-DANN (bottleneck)               | 0.2744 | 0.2793 | 0.2735 | 0.2745 | 0.2707 |        0.86 |
| Query distillation score (bottleneck) | 0.2382 | 0.2375 | 0.2368 | 0.2333 | 0.2365 |        0.50 |
| Query distillation score (linear)     | 0.2363 | 0.2367 | 0.2360 | 0.2361 | 0.2365 |        0.06 |
| TSDAE-SAPT (task queries), final      | 0.1520 | 0.1442 | 0.1503 | 0.1484 | 0.1476 |        0.78 |

## 4. Formal nDCG@10 per seed

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query InfoNCE                         | 0.4154 | 0.4109 | 0.4135 | 0.4125 | 0.4102 |        0.52 |
| Query-DANN (linear)                   | 0.3947 | 0.3954 | 0.3954 | 0.3948 | 0.3942 |        0.12 |
| Query-DANN (bottleneck)               | 0.3587 | 0.3716 | 0.3641 | 0.3594 | 0.3623 |        1.29 |
| Query distillation score (bottleneck) | 0.3150 | 0.3116 | 0.3163 | 0.3089 | 0.3127 |        0.74 |
| Query distillation score (linear)     | 0.3211 | 0.3209 | 0.3207 | 0.3206 | 0.3210 |        0.06 |
| TSDAE-SAPT (task queries), final      | 0.2517 | 0.2481 | 0.2435 | 0.2355 | 0.2543 |        1.88 |

## 5. All metrics, informal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| multilingual-mpnet base (zero-shot)   |          0.1879 |          0.2365 |          0.4728 |
| Query InfoNCE                         | 0.3553 (0.0022) | 0.4477 (0.0025) | 0.6911 (0.0020) |
| LLM rewrite                           |          0.3096 |          0.3950 |          0.6508 |
| Query-DANN (linear)                   | 0.2806 (0.0008) | 0.3528 (0.0014) | 0.5899 (0.0025) |
| Query-DANN (bottleneck)               | 0.2745 (0.0031) | 0.3446 (0.0057) | 0.5903 (0.0072) |
| Query distillation score (bottleneck) | 0.2365 (0.0019) | 0.2971 (0.0035) | 0.5381 (0.0023) |
| Query distillation score (linear)     | 0.2363 (0.0003) | 0.2996 (0.0004) | 0.5277 (0.0003) |
| TSDAE-SAPT (task queries), final      | 0.1485 (0.0029) | 0.1971 (0.0039) | 0.4178 (0.0053) |

## 6. All metrics, formal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| multilingual-mpnet base (zero-shot)   |          0.3092 |          0.3932 |          0.6528 |
| Query InfoNCE                         | 0.4125 (0.0021) | 0.5172 (0.0057) | 0.7631 (0.0030) |
| LLM rewrite                           |          0.3133 |          0.3952 |          0.6552 |
| Query-DANN (linear)                   | 0.3949 (0.0005) | 0.5030 (0.0012) | 0.7341 (0.0016) |
| Query-DANN (bottleneck)               | 0.3632 (0.0051) | 0.4600 (0.0100) | 0.7179 (0.0043) |
| Query distillation score (bottleneck) | 0.3129 (0.0029) | 0.3962 (0.0032) | 0.6536 (0.0031) |
| Query distillation score (linear)     | 0.3208 (0.0002) | 0.4062 (0.0002) | 0.6637 (0.0002) |
| TSDAE-SAPT (task queries), final      | 0.2467 (0.0074) | 0.3234 (0.0092) | 0.5554 (0.0107) |

## 7. Checkpoint step per seed

Optimiser step of the scored model: the best dev checkpoint unless the notes say otherwise. A step
that moves a lot between seeds means the dev selection itself is unstable.

| Model                                 |   42 |   84 |  123 |  256 |  512 |
|---------------------------------------|-----:|-----:|-----:|-----:|-----:|
| Query InfoNCE                         | 1150 | 1150 | 1300 | 1100 | 1150 |
| Query-DANN (linear)                   | 1350 |  950 | 1350 |  950 | 1150 |
| Query-DANN (bottleneck)               | 1000 |  550 |  950 | 1150 |  650 |
| Query distillation score (bottleneck) |  450 |  700 |  900 |  700 | 1150 |
| Query distillation score (linear)     | 1050 | 1250 |  950 | 1350 | 1150 |
| TSDAE-SAPT (task queries), final      | 1350 | 1350 | 1350 | 1350 | 1350 |

## Notes

- Seeds: Only the training seed changes between runs (initialisation, batch order, sampled hard
  negatives, TSDAE noise, dropout); queries, split, caches and protocol are identical. Seed 42 is
  the run in report_test_mpnet_official_varied.md. Base and LLM rewrite involve no training: scored
  once, so they have no std.
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
