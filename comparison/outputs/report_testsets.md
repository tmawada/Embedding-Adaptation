# Extra test sets: BGE-M3

- Encoder:   BGE-M3 (BAAI/bge-m3)
- Models:    the reported checkpoints, not retrained; every trained model once per seed (42, 84, 123, 256, 512)
- Scores:    nDCG@10 unless stated otherwise; pts = nDCG@10 x 100; gains against BGE-M3 base on
             the same queries; * = p < 0.05 (bootstrap, 1000 samples)
- Built by:  comparison/prepare_testset.py; scored by comparison/evaluate_all.py --testset

## A. Style shift: do the gains hold for a different informal style?

Data: the 83 test queries that also have a varied-style informal rewrite
(../BGE_M3/data/pairs/new_pairs.csv). Same questions, same formal text, two informal versions:
template = the style of every training query (train_pairs.csv: lower-case, question word moved to
the end, mostly '... sih'); varied = a rewrite whose style is sampled per query (slang,
abbreviations, typos, word order, capitalisation, particles).

### A1. Base model on the three versions

| Version            | nDCG@10 | MRR@10 | Recall@100 | vs formal (pts) |
|--------------------|--------:|-------:|-----------:|----------------:|
| Formal             |  0.5946 | 0.7430 |     0.9287 |               - |
| Informal, template |  0.5421 | 0.6936 |     0.8735 |          -5.25* |
| Informal, varied   |  0.4754 | 0.6157 |     0.8105 |         -11.92* |

Varied minus template on base: -6.67* pts [-12.11, -1.12], p 0.018. Negative = the varied style is
harder for BGE-M3 than the template style.

### A2. Gain over base per informal style

Gain = mean difference from base over seeds in pts (std across seeds in brackets). Sig. = seeds
significantly better / worse than base on their own (paired bootstrap over the queries, p < 0.05).
Gap closed T / V = share of base's formal-informal gap of that style recovered, mean over seeds.

| Model                                 | n | Template gain | Sig. |  Varied gain | Sig. | Gap closed T | Gap closed V |
|---------------------------------------|--:|--------------:|-----:|-------------:|-----:|-------------:|-------------:|
| Query distillation score (linear)     | 5 |  +3.15 (0.16) |  5/0 | +4.28 (0.06) |  5/0 |        60.0% |        35.9% |
| Query distillation score (bottleneck) | 5 |  +2.61 (0.43) |  1/0 | +4.15 (0.88) |  4/0 |        49.7% |        34.8% |
| Query-DANN (linear)                   | 5 |  +1.98 (0.36) |  1/0 | +3.76 (0.16) |  5/0 |        37.7% |        31.5% |
| Query-DANN (bottleneck)               | 5 |  +1.33 (0.50) |  0/0 | +1.93 (0.59) |  0/0 |        25.4% |        16.2% |

### A3. Does the gain shrink under the varied style?

Change = varied gain minus template gain, in pts, per question. Change, 95% CI and p are for the
seed-averaged model (per-query scores averaged over seeds), paired bootstrap over the 83 questions;
Seeds up/down = single seeds whose change is significant on its own. Change near 0 = the gain
carries over to the new style; clearly negative = part of the gain is tied to the training template.

| Model                                 | Change (pts) |         95% CI |      p | Seeds up/down |
|---------------------------------------|-------------:|---------------:|-------:|--------------:|
| Query distillation score (linear)     |       +1.13  | [-2.01, +4.47] | 0.497  |           0/0 |
| Query distillation score (bottleneck) |       +1.54  | [-1.78, +5.05] | 0.397  |           0/0 |
| Query-DANN (linear)                   |       +1.77  | [-1.54, +4.90] | 0.301  |           0/0 |
| Query-DANN (bottleneck)               |       +0.60  | [-2.60, +3.77] | 0.700  |           0/0 |

## B. Contamination: formal queries the encoder has seen vs never seen

Seen = our 553 test queries. They come from the official MIRACL-id train split, which BGE-M3 was
fine-tuned on together with their relevant passages. Unseen = 867 official MIRACL-id dev queries
(data/miracl_dev/, from data/topics.miracl-v1.0-id-dev.tsv), never used to train the encoders.
Formal queries only: the dev queries have no informal version yet. Both sets search the same 500,000
passages.

### B1. Base models: drop from seen to unseen queries

Drop = seen minus unseen nDCG@10, in pts; Relative = the drop as a share of the seen score. 95% CI
and p from a bootstrap that resamples each query set on its own (the sets differ, so it is
unpaired).

| Encoder                        |   Seen | Unseen | Drop (pts) |         95% CI | Relative |         95% CI |
|--------------------------------|-------:|-------:|-----------:|---------------:|---------:|---------------:|
| BGE-M3                         | 0.6141 | 0.6053 |     +0.88  | [-2.39, +4.31] |     1.4% |  [-3.9%, 6.8%] |
| multilingual-mpnet (yardstick) | 0.2935 | 0.3092 |     -1.57  | [-4.66, +1.74] |    -5.4% | [-17.2%, 5.5%] |

Memory effect = BGE-M3 drop minus multilingual-mpnet drop: +2.45 pts [-0.86, +5.86], p 0.160;
relative: +6.8 percentage points [-2.6%, 17.1%], p 0.198. multilingual-mpnet was never trained on
MIRACL, so its drop shows how much harder the unseen queries are by themselves; the extra drop of
BGE-M3 is what training on the seen queries added. Compare the relative column too:
multilingual-mpnet scores far lower, so the same difficulty costs it fewer absolute points.

### B2. All metrics, base models

| Encoder                        |      Queries | nDCG@10 | MRR@10 | Recall@100 |
|--------------------------------|-------------:|--------:|-------:|-----------:|
| BGE-M3                         |   seen (553) |  0.6141 | 0.7196 |     0.9375 |
|                                | unseen (867) |  0.6053 | 0.7048 |     0.9129 |
| multilingual-mpnet (yardstick) |   seen (553) |  0.2935 | 0.3711 |     0.6561 |
|                                | unseen (867) |  0.3092 | 0.3932 |     0.6528 |

### B3. Trained models on formal queries: seen vs unseen

Change = mean nDCG@10 difference from base over seeds in pts (std across seeds in brackets), on the
formal queries. Sig. = seeds significantly better / worse than base. About 0 on both = formal
retrieval stays intact; a change only on the unseen queries = the preservation does not carry over
to new questions.

| Model                                 | n |  Seen change | Sig. | Unseen change | Sig. |
|---------------------------------------|--:|-------------:|-----:|--------------:|-----:|
| Query-DANN (bottleneck)               | 5 | -0.20 (0.44) |  0/0 |  -0.23 (0.36) |  0/1 |
| Query-DANN (linear)                   | 5 | +0.00 (0.06) |  0/0 |  -0.06 (0.13) |  0/0 |
| Query distillation score (bottleneck) | 5 | +0.04 (0.20) |  0/0 |  -0.33 (0.09) |  0/2 |
| Query distillation score (linear)     | 5 | +0.08 (0.09) |  0/0 |  -0.47 (0.05) |  0/5 |

## Notes

- Seeds: the same five training runs per model as report_std_test.md; nothing was retrained. Base
  involves no training and is scored once.
- Query-DANN (bottleneck): the default adapter: zero-initialised residual bottleneck 1024 -> 128 ->
  1024 on the frozen BGE-M3 query vector. Same adapter in "Query distillation score (bottleneck)".
- Query-DANN (linear): --adapter linear: a single 1024 x 1024 linear map initialised to the
  identity. Same adapter in "Query distillation score (linear)".

## Missing rows

- LLM rewrite (style shift): no LLM rewrites of these queries (they would need API calls)
- TSDAE-SAPT (task queries), final (style shift): seeds 42, 84, 123, 256, 512: its fine-tuned query
  encoders are gone (methods/tsdae_sapt_queries/outputs/)
- Query InfoNCE (style shift): seeds 42, 84, 123, 256, 512: its fine-tuned query encoders are gone
  (methods/query_infonce/outputs/)
- LLM rewrite (seen vs unseen): no LLM rewrites of these queries (they would need API calls)
- TSDAE-SAPT (task queries), final (seen vs unseen): seeds 42, 84, 123, 256, 512: its fine-tuned
  query encoders are gone (methods/tsdae_sapt_queries/outputs/)
- Query InfoNCE (seen vs unseen): seeds 42, 84, 123, 256, 512: its fine-tuned query encoders are
  gone (methods/query_infonce/outputs/)
