# Seed stability: MIRACL-id test set

- Encoder:   BGE-M3 (BAAI/bge-m3)
- Queries:   867 test queries, each in a formal and an informal version
- Dataset:   data/official: trained on official MIRACL-id train, checkpoint on our dev split, tested
             on official MIRACL-id dev queries (never used to train the encoders)
- Informal:  template style (the style of every training query, mostly '... sih')
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
| BGE-M3 base (zero-shot)               | - | 0.5626 |          - |         - |          - |            - |
| LLM rewrite                           | - | 0.6007 |      +3.80 |         - |      89.2% |            - |
| Query distillation score (linear)     | 5 | 0.5809 |      +1.83 |      0.05 |      42.8% |          5/0 |
| Query distillation score (bottleneck) | 5 | 0.5803 |      +1.76 |      0.18 |      41.4% |          5/0 |
| Query InfoNCE                         | 5 | 0.5785 |      +1.59 |      0.31 |      37.3% |          4/0 |
| Query-DANN (linear)                   | 5 | 0.5734 |      +1.08 |      0.16 |      25.4% |          5/0 |
| Query-DANN (bottleneck)               | 5 | 0.5692 |      +0.65 |      0.27 |      15.3% |          2/0 |
| TSDAE-SAPT (task queries), final      | 5 | 0.4466 |     -11.61 |      1.20 |    -272.3% |          0/5 |

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
| Query distillation score (linear)     | 0.5802 | 0.5810 | 0.5806 | 0.5812 | 0.5815 |        0.13 |
| Query distillation score (bottleneck) | 0.5794 | 0.5806 | 0.5812 | 0.5777 | 0.5825 |        0.48 |
| Query InfoNCE                         | 0.5792 | 0.5819 | 0.5776 | 0.5803 | 0.5737 |        0.82 |
| Query-DANN (linear)                   | 0.5715 | 0.5745 | 0.5752 | 0.5721 | 0.5739 |        0.37 |
| Query-DANN (bottleneck)               | 0.5685 | 0.5667 | 0.5716 | 0.5667 | 0.5724 |        0.58 |
| TSDAE-SAPT (task queries), final      | 0.4670 | 0.4399 | 0.4433 | 0.4367 | 0.4459 |        3.03 |

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
| BGE-M3 base (zero-shot)               |          0.5626 |          0.6719 |          0.8870 |
| LLM rewrite                           |          0.6007 |          0.6995 |          0.9154 |
| Query distillation score (linear)     | 0.5809 (0.0005) | 0.6826 (0.0014) | 0.9050 (0.0006) |
| Query distillation score (bottleneck) | 0.5803 (0.0018) | 0.6825 (0.0019) | 0.9043 (0.0035) |
| Query InfoNCE                         | 0.5785 (0.0031) | 0.6788 (0.0034) | 0.9013 (0.0025) |
| Query-DANN (linear)                   | 0.5734 (0.0016) | 0.6779 (0.0021) | 0.9019 (0.0019) |
| Query-DANN (bottleneck)               | 0.5692 (0.0027) | 0.6732 (0.0061) | 0.8968 (0.0038) |
| TSDAE-SAPT (task queries), final      | 0.4466 (0.0120) | 0.5557 (0.0102) | 0.7997 (0.0141) |

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
  the run in report_test_official.md. Base and LLM rewrite involve no training: scored once, so they
  have no std.
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
