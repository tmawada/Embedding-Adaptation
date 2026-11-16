# Harmonized comparison: MIRACL-id test set

- Queries:   578 test queries, each in a formal and an informal version
- Passages:  500,000, one frozen BGE-M3 index searched by every row
- Scores:    nDCG@10 unless stated otherwise; pts = nDCG@10 x 100

## Training protocol (every trained model)

- Steps:       1350 optimiser steps, effective batch 64
- Optimiser:   AdamW, lr 3e-05, weight decay 0.01, grad clip 1.0
- Schedule:    cosine down to 1e-06 after 10% linear warmup
- Precision:   bf16, max_length 128 (encoder fine-tuning)
- Seed:        42
- Checkpoint:  best dev informal nDCG@10, checked every 50 steps

## 1. Summary

Sorted by informal nDCG@10. Gain and Change = difference from base in pts (* = p < 0.05, paired
bootstrap, 1000 samples). Gap closed = share of the base model's formal-informal gap recovered on
informal queries.

| Model                            | Informal | Gain (pts) | Formal | Change (pts) | Gap closed |
|----------------------------------|---------:|-----------:|-------:|-------------:|-----------:|
| BGE-M3 base (zero-shot)          |   0.5373 |         -  | 0.6166 |           -  |          - |
| LLM rewrite                      |   0.6017 |     +6.44* | 0.5966 |       -2.00* |      81.2% |
| Query distillation (score)       |   0.5740 |     +3.66* | 0.6172 |       +0.05  |      46.2% |
| Stacked (TSDAE + DANN + KD)      |   0.5716 |     +3.43* | 0.6162 |       -0.04  |      43.2% |
| Embedding transform (linear)     |   0.5676 |     +3.03* | 0.6017 |       -1.49* |      38.2% |
| UDA Query-DANN                   |   0.5578 |     +2.05* | 0.6037 |       -1.29* |      25.9% |
| Query InfoNCE + TSDAE            |   0.5563 |     +1.90  | 0.5844 |       -3.22* |      24.0% |
| UDA Query-DANN (strict)          |   0.5553 |     +1.80* | 0.6050 |       -1.16* |      22.7% |
| Query-DANN                       |   0.5549 |     +1.76* | 0.6171 |       +0.05  |      22.2% |
| TSDAE-SAPT (task queries)        |   0.5363 |     -0.10  | 0.6176 |       +0.09  |      -1.3% |
| TSDAE-SAPT (reviews)             |   0.5309 |     -0.64* | 0.6159 |       -0.07  |      -8.1% |
| TSDAE-SAPT (task queries), final |   0.4472 |     -9.01* | 0.5272 |       -8.95* |    -113.6% |
| TSDAE-SAPT (reviews), final      |   0.3563 |    -18.10* | 0.4803 |      -13.63* |    -228.2% |

## 2. All metrics, informal queries

| Model                            | nDCG@10 | MRR@10 | Recall@100 |
|----------------------------------|--------:|-------:|-----------:|
| BGE-M3 base (zero-shot)          |  0.5373 | 0.6391 |     0.9003 |
| LLM rewrite                      |  0.6017 | 0.7101 |     0.9369 |
| Query distillation (score)       |  0.5740 | 0.6769 |     0.9128 |
| Stacked (TSDAE + DANN + KD)      |  0.5716 | 0.6765 |     0.9107 |
| Embedding transform (linear)     |  0.5676 | 0.6819 |     0.8991 |
| UDA Query-DANN                   |  0.5578 | 0.6655 |     0.9064 |
| Query InfoNCE + TSDAE            |  0.5563 | 0.6688 |     0.8976 |
| UDA Query-DANN (strict)          |  0.5553 | 0.6606 |     0.9074 |
| Query-DANN                       |  0.5549 | 0.6644 |     0.8992 |
| TSDAE-SAPT (task queries)        |  0.5363 | 0.6383 |     0.8991 |
| TSDAE-SAPT (reviews)             |  0.5309 | 0.6393 |     0.8973 |
| TSDAE-SAPT (task queries), final |  0.4472 | 0.5458 |     0.8219 |
| TSDAE-SAPT (reviews), final      |  0.3563 | 0.4561 |     0.7264 |

## 3. All metrics, formal queries

| Model                            | nDCG@10 | MRR@10 | Recall@100 |
|----------------------------------|--------:|-------:|-----------:|
| BGE-M3 base (zero-shot)          |  0.6166 | 0.7248 |     0.9393 |
| LLM rewrite                      |  0.5966 | 0.7051 |     0.9327 |
| Query distillation (score)       |  0.6172 | 0.7212 |     0.9417 |
| Stacked (TSDAE + DANN + KD)      |  0.6162 | 0.7230 |     0.9417 |
| Embedding transform (linear)     |  0.6017 | 0.7173 |     0.9321 |
| UDA Query-DANN                   |  0.6037 | 0.7042 |     0.9408 |
| Query InfoNCE + TSDAE            |  0.5844 | 0.6948 |     0.9222 |
| UDA Query-DANN (strict)          |  0.6050 | 0.7058 |     0.9383 |
| Query-DANN                       |  0.6171 | 0.7270 |     0.9412 |
| TSDAE-SAPT (task queries)        |  0.6176 | 0.7253 |     0.9428 |
| TSDAE-SAPT (reviews)             |  0.6159 | 0.7239 |     0.9406 |
| TSDAE-SAPT (task queries), final |  0.5272 | 0.6392 |     0.8777 |
| TSDAE-SAPT (reviews), final      |  0.4803 | 0.5928 |     0.8590 |

## 4. Significance vs base and selected checkpoint

nDCG@10 difference from base in pts, paired bootstrap with 1000 samples; * = p < 0.05. Step =
optimiser step of the dev-selected checkpoint (- = not trained).

| Model                            | Informal gain |           95% CI |      p | Formal change |      p | Step |
|----------------------------------|--------------:|-----------------:|-------:|--------------:|-------:|-----:|
| LLM rewrite                      |        +6.44* |   [+4.74, +8.12] | 0.001* |        -2.00* | 0.002* |    - |
| Query distillation (score)       |        +3.66* |   [+2.56, +4.76] | 0.001* |        +0.05  | 0.804  |  950 |
| Stacked (TSDAE + DANN + KD)      |        +3.43* |   [+2.17, +4.59] | 0.001* |        -0.04  | 0.906  | 1350 |
| Embedding transform (linear)     |        +3.03* |   [+1.69, +4.30] | 0.001* |        -1.49* | 0.001* |    - |
| UDA Query-DANN                   |        +2.05* |   [+0.86, +3.14] | 0.001* |        -1.29* | 0.011* |  700 |
| Query InfoNCE + TSDAE            |        +1.90  |   [-0.13, +3.70] | 0.052  |        -3.22* | 0.001* | 1350 |
| UDA Query-DANN (strict)          |        +1.80* |   [+0.64, +2.93] | 0.006* |        -1.16* | 0.028* |  850 |
| Query-DANN                       |        +1.76* |   [+0.38, +3.14] | 0.006* |        +0.05  | 0.917  | 1350 |
| TSDAE-SAPT (task queries)        |        -0.10  |   [-0.35, +0.15] | 0.405  |        +0.09  | 0.207  |   50 |
| TSDAE-SAPT (reviews)             |        -0.64* |   [-1.11, -0.20] | 0.009* |        -0.07  | 0.711  |   50 |
| TSDAE-SAPT (task queries), final |        -9.01* |  [-10.91, -7.01] | 0.001* |        -8.95* | 0.001* | 1350 |
| TSDAE-SAPT (reviews), final      |       -18.10* | [-20.26, -15.86] | 0.001* |       -13.63* | 0.001* | 1350 |

## Notes

- LLM rewrite: gemini-3.5-flash-lite, cached rewrites of the test queries (no API calls); not
  trained. Its formal column rewrites the formal queries too, since a deployed rewriter sees every
  query.
- Stacked (TSDAE + DANN + KD): Query-DANN + score-distillation adapter trained on top of the
  TSDAE-SAPT (task queries) encoder.
- Embedding transform (linear): closed-form least squares fitted on the train split only; no
  optimiser, so no step budget.
- UDA Query-DANN: checkpoint selected on dev informal nDCG@10 like every other row, so it uses
  target-domain dev labels.
- UDA Query-DANN (strict): supplementary; checkpoint selected on dev formal queries passed through
  the adapter, no informal labels (strict UDA).
- TSDAE-SAPT (task queries): dev-selected checkpoint (see Step); "TSDAE-SAPT (task queries), final"
  is the same training run after its last step.
- TSDAE-SAPT (reviews): dev-selected checkpoint (see Step); "TSDAE-SAPT (reviews), final" is the
  same training run after its last step.
- TSDAE-SAPT (task queries), final: supplementary; the encoder after the last training step, without
  dev selection, from the same run as "TSDAE-SAPT (task queries)". It shows what the training itself
  does to retrieval.
- TSDAE-SAPT (reviews), final: supplementary; the encoder after the last training step, without dev
  selection, from the same run as "TSDAE-SAPT (reviews)". It shows what the training itself does to
  retrieval.
