# Seed stability: MIRACL-id test set

- Encoder:   multilingual-mpnet (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- Queries:   867 test queries, each in a formal and an informal version
- Dataset:   data/official: trained on official MIRACL-id train, checkpoint on our dev split, tested
             on official MIRACL-id dev queries (never used to train the encoders)
- Informal:  template style (the style of every training query, mostly '... sih')
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
| multilingual-mpnet base (zero-shot)   | - | 0.2226 |          - |         - |          - |            - |
| Query InfoNCE                         | 5 | 0.3963 |     +17.37 |      0.49 |     200.6% |          5/0 |
| Query-DANN (linear)                   | 5 | 0.3213 |      +9.87 |      0.08 |     113.9% |          5/0 |
| Query-DANN (bottleneck)               | 5 | 0.3207 |      +9.81 |      0.13 |     113.2% |          5/0 |
| LLM rewrite                           | - | 0.3080 |      +8.53 |         - |      98.6% |            - |
| Query distillation score (bottleneck) | 5 | 0.2671 |      +4.44 |      0.28 |      51.3% |          5/0 |
| Query distillation score (linear)     | 5 | 0.2635 |      +4.09 |      0.03 |      47.2% |          5/0 |
| TSDAE-SAPT (task queries), final      | 5 | 0.1718 |      -5.08 |      0.44 |     -58.7% |          0/5 |

## 2. Formal queries: mean over seeds and stability

Did training hurt formal queries? Change = mean difference from base in pts; Std and Sig. up/down as
in section 1.

| Model                                 | n |   Mean | Change (pts) | Std (pts) | Sig. up/down |
|---------------------------------------|--:|-------:|-------------:|----------:|-------------:|
| multilingual-mpnet base (zero-shot)   | - | 0.3092 |            - |         - |            - |
| Query InfoNCE                         | 5 | 0.4125 |       +10.33 |      0.21 |          5/0 |
| Query-DANN (linear)                   | 5 | 0.3949 |        +8.57 |      0.05 |          5/0 |
| Query-DANN (bottleneck)               | 5 | 0.3632 |        +5.40 |      0.51 |          5/0 |
| LLM rewrite                           | - | 0.3133 |        +0.41 |         - |            - |
| Query distillation score (bottleneck) | 5 | 0.3129 |        +0.37 |      0.29 |          1/0 |
| Query distillation score (linear)     | 5 | 0.3208 |        +1.16 |      0.02 |          5/0 |
| TSDAE-SAPT (task queries), final      | 5 | 0.2467 |        -6.26 |      0.74 |          0/5 |

## 3. Informal nDCG@10 per seed

Range = best minus worst seed, in pts.

| Model                                 |     42 |     84 |    123 |    256 |    512 | Range (pts) |
|---------------------------------------|-------:|-------:|-------:|-------:|-------:|------------:|
| Query InfoNCE                         | 0.4009 | 0.3987 | 0.3993 | 0.3934 | 0.3892 |        1.17 |
| Query-DANN (linear)                   | 0.3216 | 0.3215 | 0.3223 | 0.3204 | 0.3206 |        0.19 |
| Query-DANN (bottleneck)               | 0.3217 | 0.3199 | 0.3218 | 0.3212 | 0.3188 |        0.30 |
| Query distillation score (bottleneck) | 0.2658 | 0.2702 | 0.2687 | 0.2629 | 0.2678 |        0.73 |
| Query distillation score (linear)     | 0.2638 | 0.2635 | 0.2631 | 0.2637 | 0.2633 |        0.07 |
| TSDAE-SAPT (task queries), final      | 0.1779 | 0.1664 | 0.1738 | 0.1718 | 0.1692 |        1.16 |

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
| multilingual-mpnet base (zero-shot)   |          0.2226 |          0.2810 |          0.5334 |
| Query InfoNCE                         | 0.3963 (0.0049) | 0.4986 (0.0067) | 0.7489 (0.0032) |
| Query-DANN (linear)                   | 0.3213 (0.0008) | 0.4086 (0.0013) | 0.6493 (0.0010) |
| Query-DANN (bottleneck)               | 0.3207 (0.0013) | 0.4094 (0.0016) | 0.6549 (0.0036) |
| LLM rewrite                           |          0.3080 |          0.3856 |          0.6513 |
| Query distillation score (bottleneck) | 0.2671 (0.0028) | 0.3372 (0.0039) | 0.5963 (0.0022) |
| Query distillation score (linear)     | 0.2635 (0.0003) | 0.3302 (0.0003) | 0.5909 (0.0007) |
| TSDAE-SAPT (task queries), final      | 0.1718 (0.0044) | 0.2272 (0.0058) | 0.4427 (0.0085) |

## 6. All metrics, formal queries: mean (std) over seeds

| Model                                 |         nDCG@10 |          MRR@10 |      Recall@100 |
|---------------------------------------|----------------:|----------------:|----------------:|
| multilingual-mpnet base (zero-shot)   |          0.3092 |          0.3932 |          0.6528 |
| Query InfoNCE                         | 0.4125 (0.0021) | 0.5172 (0.0057) | 0.7631 (0.0030) |
| Query-DANN (linear)                   | 0.3949 (0.0005) | 0.5030 (0.0012) | 0.7341 (0.0016) |
| Query-DANN (bottleneck)               | 0.3632 (0.0051) | 0.4600 (0.0100) | 0.7179 (0.0043) |
| LLM rewrite                           |          0.3133 |          0.3952 |          0.6552 |
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
  the run in report_test_mpnet_official.md. Base and LLM rewrite involve no training: scored once,
  so they have no std.
- Query InfoNCE: --objective infonce: the full encoder fine-tuned with InfoNCE on informal queries
  against the frozen passage index, without the TSDAE reconstruction term that the method adds by
  default (weight 0.1).
- Query-DANN (linear): --adapter linear: a single 768 x 768 linear map initialised to the identity.
  Same adapter in "Query distillation score (linear)".
- Query-DANN (bottleneck): the default adapter: zero-initialised residual bottleneck 768 -> 128 ->
  768 on the frozen multilingual-mpnet query vector. Same adapter in "Query distillation score
  (bottleneck)".
- LLM rewrite: gemini-3.5-flash-lite, cached rewrites of the test queries (no API calls); not
  trained. Its formal column rewrites the formal queries too, since a deployed rewriter sees every
  query.
- TSDAE-SAPT (task queries), final: the encoder after the last training step, NOT the best dev
  checkpoint like the other rows: dev selection keeps step 50, which has barely trained and scores
  like base. This row shows what TSDAE training itself does to retrieval.
