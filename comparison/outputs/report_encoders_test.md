# Encoder comparison: MIRACL-id test set

- BGE-M3:    BGE-M3 (BAAI/bge-m3)
- mE5:       mE5-large-instruct (intfloat/multilingual-e5-large-instruct)
- mpnet:     multilingual-mpnet (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- Arctic:    Arctic-Embed-L-2.0 (Snowflake/snowflake-arctic-embed-l-v2.0)
- Queries:   553 test queries, each in a formal and an informal version
- Passages:  each encoder searches its own frozen index of the same 500,000 passages
- Scores:    nDCG@10; pts = nDCG@10 x 100; every gain is measured against THAT encoder's own base

## Training protocol (every trained model)

- Steps:       1350 optimiser steps, effective batch 64
- Optimiser:   AdamW, lr 3e-05, weight decay 0.01, grad clip 1.0
- Schedule:    cosine down to 1e-06 after 10% linear warmup
- Precision:   bf16, max_length 128 (encoder fine-tuning)
- Seed:        42
- Checkpoint:  best dev informal nDCG@10, checked every 50 steps

## 1. Zero-shot robustness of the base encoders

Drop = informal minus formal nDCG@10 on the base model, in pts (* = p < 0.05, paired bootstrap, 1000
samples): the gap the methods try to close.

| Encoder            | Formal | Informal | Drop (pts) |
|--------------------|-------:|---------:|-----------:|
| BGE-M3             | 0.6141 |   0.5362 |     -7.79* |
| mE5-large-instruct | 0.5510 |   0.5097 |     -4.13* |
| multilingual-mpnet | 0.2935 |   0.1948 |     -9.87* |
| Arctic-Embed-L-2.0 | 0.5174 |   0.4936 |     -2.37* |

## 2. Informal and formal queries per model and encoder

nDCG@10; Gain and Change = difference from that encoder's base in pts (* = p < 0.05). Gap closed =
share of that encoder's formal-informal gap recovered on informal queries.

| Model                                 | Encoder | Informal |    Gain | Gap closed | Formal |  Change |
|---------------------------------------|--------:|---------:|--------:|-----------:|-------:|--------:|
| LLM rewrite                           |  BGE-M3 |   0.5983 |  +6.21* |      79.7% | 0.5926 |  -2.15* |
|                                       |     mE5 |   0.5437 |  +3.40* |      82.3% | 0.5382 |  -1.28* |
|                                       |   mpnet |   0.3015 | +10.67* |     108.1% | 0.3066 |  +1.31* |
|                                       |  Arctic |   0.5094 |  +1.58* |      66.5% | 0.5062 |  -1.11* |
| Query-DANN (bottleneck)               |  BGE-M3 |   0.5488 |  +1.26  |      16.2% | 0.6132 |  -0.09  |
|                                       |     mE5 |   0.5190 |  +0.94* |      22.6% | 0.5448 |  -0.62  |
|                                       |   mpnet |   0.2796 |  +8.47* |      85.9% | 0.3454 |  +5.19* |
|                                       |  Arctic |   0.5194 |  +2.58* |     108.7% | 0.5236 |  +0.63  |
| Query-DANN (linear)                   |  BGE-M3 |   0.5549 |  +1.88* |      24.1% | 0.6135 |  -0.05  |
|                                       |     mE5 |   0.5174 |  +0.77* |      18.6% | 0.5493 |  -0.17  |
|                                       |   mpnet |   0.2875 |  +9.26* |      93.9% | 0.3723 |  +7.88* |
|                                       |  Arctic |   0.5239 |  +3.02* |     127.5% | 0.5354 |  +1.81* |
| Query distillation score (bottleneck) |  BGE-M3 |   0.5710 |  +3.49* |      44.8% | 0.6153 |  +0.12  |
|                                       |     mE5 |   0.5256 |  +1.59* |      38.5% | 0.5485 |  -0.25  |
|                                       |   mpnet |   0.2264 |  +3.15* |      32.0% | 0.2869 |  -0.66  |
|                                       |  Arctic |   0.5042 |  +1.05* |      44.4% | 0.5112 |  -0.62* |
| Query distillation score (linear)     |  BGE-M3 |   0.5725 |  +3.64* |      46.7% | 0.6146 |  +0.05  |
|                                       |     mE5 |   0.5281 |  +1.85* |      44.7% | 0.5464 |  -0.46  |
|                                       |   mpnet |   0.2266 |  +3.18* |      32.2% | 0.2968 |  +0.33  |
|                                       |  Arctic |   0.5097 |  +1.60* |      67.5% | 0.5095 |  -0.79* |
| TSDAE-SAPT (task queries), final      |  BGE-M3 |   0.4446 |  -9.16* |    -117.5% | 0.5247 |  -8.94* |
|                                       |     mE5 |   0.2186 | -29.11* |    -704.7% | 0.2458 | -30.52* |
|                                       |   mpnet |   0.1506 |  -4.43* |     -44.8% | 0.2410 |  -5.25* |
|                                       |  Arctic |   0.4680 |  -2.56* |    -108.2% | 0.4774 |  -4.00* |
| Query InfoNCE                         |  BGE-M3 |   0.5465 |  +1.03  |      13.2% | 0.5775 |  -3.65* |
|                                       |     mE5 |   0.4856 |  -2.41* |     -58.3% | 0.5099 |  -4.11* |
|                                       |   mpnet |   0.3750 | +18.02* |     182.6% | 0.4012 | +10.77* |
|                                       |  Arctic |   0.5387 |  +4.51* |     190.0% | 0.5438 |  +2.64* |

## Notes

- Details per encoder (MRR@10, Recall@100, 95% CIs, p-values, selected steps): report_test.md,
  report_test_me5.md, report_test_mpnet.md, report_test_arctic.md.
- LLM rewrite: gemini-3.5-flash-lite, cached rewrites of the test queries (no API calls); not
  trained. Its formal column rewrites the formal queries too, since a deployed rewriter sees every
  query.
- Query-DANN (bottleneck): the default adapter: zero-initialised residual bottleneck d -> 128 -> d
  on the frozen base query vector. Same adapter in "Query distillation score (bottleneck)".
- Query-DANN (linear): --adapter linear: a single d x d linear map initialised to the identity. Same
  adapter in "Query distillation score (linear)".
- Query InfoNCE: --objective infonce: the full encoder fine-tuned with InfoNCE on informal queries
  against the frozen passage index, without the TSDAE reconstruction term that the method adds by
  default (weight 0.1).
- TSDAE-SAPT (task queries), final: the encoder after the last training step, NOT the best dev
  checkpoint like the other rows: dev selection keeps step 50, which has barely trained and scores
  like base. This row shows what TSDAE training itself does to retrieval.
- Adapter size: d = the encoder's output size (BGE-M3 1024, mE5-large-instruct 1024,
  multilingual-mpnet 768, Arctic-Embed-L-2.0 1024).
- Temperature: mE5-large-instruct's cosine scores are compressed (relevant and top-ranked
  non-relevant passages both score ~0.89, random ones ~0.76), while the InfoNCE / distillation
  temperature (tau 0.05) was set for BGE-M3. The protocol is kept identical on purpose, so mE5 may
  be at a disadvantage on those methods.
