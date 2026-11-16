# Seed stability: MIRACL-id test set

- Encoder:   multilingual-mpnet (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- Queries:   553 test queries, each in a formal and an informal version
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
| multilingual-mpnet base (zero-shot)   | - | 0.1948 |          - |         - |          - |            - |
| Query InfoNCE                         | 5 | 0.3790 |     +18.42 |      0.27 |     186.6% |          5/0 |
| LLM rewrite                           | - | 0.3015 |     +10.67 |         - |     108.1% |            - |
| Query-DANN (linear)                   | 5 | 0.2888 |      +9.39 |      0.12 |      95.2% |          5/0 |
| Query-DANN (bottleneck)               | 5 | 0.2791 |      +8.43 |      0.48 |      85.4% |          5/0 |
| Query distillation score (bottleneck) | 5 | 0.2282 |      +3.33 |      0.32 |      33.8% |          5/0 |
| Query distillation score (linear)     | 5 | 0.2266 |      +3.18 |      0.01 |      32.2% |          5/0 |
| TSDAE-SAPT (task queries), final      | 5 | 0.1497 |      -4.51 |      0.17 |     -45.7% |          0/5 |

## 2. Formal queries: mean over seeds and stability

Did training hurt formal queries? Change = mean difference from base in pts; Std and Sig. up/down as
in section 1.

| Model                                 | n |   Mean | Change (pts) | Std (pts) | Sig. up/down |
|---------------------------------------|--:|-------:|-------------:|----------:|-------------:|
| multilingual-mpnet base (zero-shot)   | - | 0.2935 |            - |         - |            - |
| Query InfoNCE                         | 5 | 0.4028 |       +10.93 |      0.14 |          5/0 |
| LLM rewrite                           | - | 0.3066 |        +1.31 |         - |            - |
| Query-DANN (linear)                   | 5 | 0.3683 |        +7.48 |      0.24 |          5/0 |
| Query-DANN (bottleneck)               | 5 | 0.3466 |        +5.31 |      0.67 |          5/0 |
| Query distillation score (bottleneck) | 5 | 0.2896 |        -0.39 |      0.34 |          0/0 |
| Query distillation score (linear)     | 5 | 0.2967 |        +0.31 |      0.04 |          0/0 |
| TSDAE-SAPT (task queries), final      | 5 | 0.2340 |        -5.95 |      1.03 |          0/5 |

## 3. Informal nDCG@10 per seed

Range = best minus worst seed, in pts.

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query InfoNCE                         | 0.3750 | 0.3816 | 0.3775 | 0.3801 | 0.3808 |        0.66 |
| Query-DANN (linear)                   | 0.2875 | 0.2886 | 0.2891 | 0.2881 | 0.2905 |        0.31 |
| Query-DANN (bottleneck)               | 0.2796 | 0.2719 | 0.2774 | 0.2824 | 0.2842 |        1.23 |
| Query distillation score (bottleneck) | 0.2264 | 0.2237 | 0.2286 | 0.2308 | 0.2314 |        0.77 |
| Query distillation score (linear)     | 0.2266 | 0.2268 | 0.2264 | 0.2267 | 0.2266 |        0.04 |
| TSDAE-SAPT (task queries), final      | 0.1506 | 0.1473 | 0.1506 | 0.1514 | 0.1487 |        0.41 |

## 4. Formal nDCG@10 per seed

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query InfoNCE                         | 0.4012 | 0.4026 | 0.4016 | 0.4038 | 0.4046 |        0.34 |
| Query-DANN (linear)                   | 0.3723 | 0.3659 | 0.3678 | 0.3674 | 0.3683 |        0.65 |
| Query-DANN (bottleneck)               | 0.3454 | 0.3450 | 0.3572 | 0.3386 | 0.3469 |        1.86 |
| Query distillation score (bottleneck) | 0.2869 | 0.2873 | 0.2898 | 0.2954 | 0.2884 |        0.84 |
| Query distillation score (linear)     | 0.2968 | 0.2959 | 0.2969 | 0.2968 | 0.2969 |        0.10 |
| TSDAE-SAPT (task queries), final      | 0.2410 | 0.2454 | 0.2326 | 0.2184 | 0.2324 |        2.70 |

## 5. All metrics, informal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| multilingual-mpnet base (zero-shot)   |          0.1948 |          0.2483 |          0.5001 |
| Query InfoNCE                         | 0.3790 (0.0027) | 0.4830 (0.0065) | 0.7231 (0.0053) |
| LLM rewrite                           |          0.3015 |          0.3755 |          0.6548 |
| Query-DANN (linear)                   | 0.2888 (0.0012) | 0.3711 (0.0014) | 0.6362 (0.0011) |
| Query-DANN (bottleneck)               | 0.2791 (0.0048) | 0.3581 (0.0047) | 0.6400 (0.0101) |
| Query distillation score (bottleneck) | 0.2282 (0.0032) | 0.2945 (0.0051) | 0.5703 (0.0058) |
| Query distillation score (linear)     | 0.2266 (0.0001) | 0.2941 (0.0003) | 0.5723 (0.0004) |
| TSDAE-SAPT (task queries), final      | 0.1497 (0.0017) | 0.1966 (0.0030) | 0.4457 (0.0152) |

## 6. All metrics, formal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| multilingual-mpnet base (zero-shot)   |          0.2935 |          0.3711 |          0.6561 |
| Query InfoNCE                         | 0.4028 (0.0014) | 0.5070 (0.0046) | 0.7658 (0.0025) |
| LLM rewrite                           |          0.3066 |          0.3822 |          0.6701 |
| Query-DANN (linear)                   | 0.3683 (0.0024) | 0.4705 (0.0033) | 0.7385 (0.0016) |
| Query-DANN (bottleneck)               | 0.3466 (0.0067) | 0.4343 (0.0124) | 0.7308 (0.0020) |
| Query distillation score (bottleneck) | 0.2896 (0.0034) | 0.3695 (0.0043) | 0.6570 (0.0016) |
| Query distillation score (linear)     | 0.2967 (0.0004) | 0.3818 (0.0009) | 0.6623 (0.0003) |
| TSDAE-SAPT (task queries), final      | 0.2340 (0.0103) | 0.3072 (0.0109) | 0.5668 (0.0175) |

## 7. Checkpoint step per seed

Optimiser step of the scored model: the best dev checkpoint unless the notes say otherwise. A step
that moves a lot between seeds means the dev selection itself is unstable.

| Model                                 |   42 |   84 |  123 |  256 |  512 |
|---------------------------------------|-----:|-----:|-----:|-----:|-----:|
| Query InfoNCE                         | 1050 | 1350 | 1350 | 1250 | 1350 |
| Query-DANN (linear)                   |  900 | 1200 |  900 | 1350 | 1250 |
| Query-DANN (bottleneck)               |  600 |  500 |  300 |  950 |  700 |
| Query distillation score (bottleneck) |  450 |  200 |  700 |  250 | 1350 |
| Query distillation score (linear)     | 1100 | 1050 | 1050 | 1100 | 1250 |
| TSDAE-SAPT (task queries), final      | 1350 | 1350 | 1350 | 1350 | 1350 |

## Notes

- Seeds: Only the training seed changes between runs (initialisation, batch order, sampled hard
  negatives, TSDAE noise, dropout); queries, split, caches and protocol are identical. Seed 42 is
  the run in report_test_mpnet.md. Base and LLM rewrite involve no training: scored once, so they
  have no std.
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
