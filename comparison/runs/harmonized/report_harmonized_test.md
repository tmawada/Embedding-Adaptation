# Retrieval results — MIRACL-id test (578 queries, 500,000 passages)

BGE-M3 dense retrieval, top-100. Harmonised training protocol for every trained model.

| Model | Formal R@100 | Formal MRR@100 | Formal nDCG@10 | Informal R@100 | Informal MRR@100 | Informal nDCG@10 |
|---|---|---|---|---|---|---|
| BGE-M3 base | 0.9393 | 0.7282 | 0.6166 | 0.9003 | 0.6441 | 0.5373 |
| BGE-M3 + LLM normalisation | 0.9327 | 0.7090 | 0.5966 | 0.9369 | 0.7131 | 0.6017 |
| Query-DANN v2 | 0.9384 | 0.7277 | 0.6136 | 0.9145 | 0.6831 | 0.5738 |
| Query distillation (score) | 0.9418 | 0.7261 | 0.6177 | 0.9164 | 0.6831 | 0.5730 |
| Query distillation (embed) | 0.9407 | 0.7330 | 0.6185 | 0.9118 | 0.6797 | 0.5652 |
| TSDAE-SAPT (reviews) | 0.9390 | 0.7222 | 0.6143 | 0.8965 | 0.6429 | 0.5351 |
| TSDAE-SAPT (reviews) + Query-DANN v2 | 0.9355 | 0.7297 | 0.6130 | 0.9110 | 0.6875 | 0.5758 |
| TSDAE-SAPT (task queries) | 0.9399 | 0.7287 | 0.6182 | 0.8969 | 0.6373 | 0.5266 |
| TSDAE-SAPT (task queries) + Query-DANN v2 | 0.9391 | 0.7233 | 0.6113 | 0.9154 | 0.6827 | 0.5754 |

## nDCG@10 formal/informal gap

Gap = informal − formal (negative = informal is worse). Narrowing vs base = how much of the base model's 7.93-point gap this pipeline removes. p tests informal vs formal within the model (paired bootstrap, 1000 samples).

| Model | Formal nDCG@10 | Informal nDCG@10 | Gap (pts) | Gap (%) | Narrowing vs base | p |
|---|---|---|---|---|---|---|
| BGE-M3 base | 0.6166 | 0.5373 | -7.93 | -12.9% | +0.00 | 0.001* |
| BGE-M3 + LLM normalisation | 0.5966 | 0.6017 | +0.51 | +0.9% | +8.44 | 0.247 |
| Query-DANN v2 | 0.6136 | 0.5738 | -3.98 | -6.5% | +3.95 | 0.001* |
| Query distillation (score) | 0.6177 | 0.5730 | -4.47 | -7.2% | +3.46 | 0.001* |
| Query distillation (embed) | 0.6185 | 0.5652 | -5.32 | -8.6% | +2.61 | 0.001* |
| TSDAE-SAPT (reviews) | 0.6143 | 0.5351 | -7.92 | -12.9% | +0.01 | 0.001* |
| TSDAE-SAPT (reviews) + Query-DANN v2 | 0.6130 | 0.5758 | -3.72 | -6.1% | +4.21 | 0.001* |
| TSDAE-SAPT (task queries) | 0.6182 | 0.5266 | -9.15 | -14.8% | -1.22 | 0.001* |
| TSDAE-SAPT (task queries) + Query-DANN v2 | 0.6113 | 0.5754 | -3.59 | -5.9% | +4.35 | 0.001* |
