# Retrieval results — MIRACL-id test (578 queries, 500,000 passages)

BGE-M3 dense retrieval, top-100. Per-method tuned settings.

| Model | Formal R@100 | Formal MRR@100 | Formal nDCG@10 | Informal R@100 | Informal MRR@100 | Informal nDCG@10 |
|---|---|---|---|---|---|---|
| BGE-M3 base | 0.9393 | 0.7282 | 0.6166 | 0.9003 | 0.6441 | 0.5373 |
| BGE-M3 + LLM normalisation | 0.9327 | 0.7090 | 0.5966 | 0.9369 | 0.7131 | 0.6017 |
| Query-DANN v2 | 0.9401 | 0.7251 | 0.6121 | 0.9084 | 0.6892 | 0.5748 |
| Query distillation (score) | 0.9434 | 0.7318 | 0.6204 | 0.9142 | 0.6864 | 0.5768 |
| Query distillation (embed) | 0.9410 | 0.7327 | 0.6193 | 0.9135 | 0.6816 | 0.5672 |
| TSDAE-SAPT (reviews) | 0.8942 | 0.6457 | 0.5340 | 0.7524 | 0.4506 | 0.3604 |
| TSDAE-SAPT (reviews) + Query-DANN v2 | 0.8910 | 0.6433 | 0.5262 | 0.8241 | 0.6025 | 0.4860 |
| TSDAE-SAPT (task queries) | 0.8958 | 0.6486 | 0.5394 | 0.8390 | 0.5544 | 0.4555 |
| TSDAE-SAPT (task queries) + Query-DANN v2 | 0.9052 | 0.6633 | 0.5530 | 0.8613 | 0.6298 | 0.5205 |

## nDCG@10 formal/informal gap

Gap = informal − formal (negative = informal is worse). Narrowing vs base = how much of the base model's 7.93-point gap this pipeline removes. p tests informal vs formal within the model (paired bootstrap, 1000 samples).

| Model | Formal nDCG@10 | Informal nDCG@10 | Gap (pts) | Gap (%) | Narrowing vs base | p |
|---|---|---|---|---|---|---|
| BGE-M3 base | 0.6166 | 0.5373 | -7.93 | -12.9% | +0.00 | 0.001* |
| BGE-M3 + LLM normalisation | 0.5966 | 0.6017 | +0.51 | +0.9% | +8.44 | 0.247 |
| Query-DANN v2 | 0.6121 | 0.5748 | -3.73 | -6.1% | +4.20 | 0.001* |
| Query distillation (score) | 0.6204 | 0.5768 | -4.35 | -7.0% | +3.58 | 0.001* |
| Query distillation (embed) | 0.6193 | 0.5672 | -5.21 | -8.4% | +2.72 | 0.001* |
| TSDAE-SAPT (reviews) | 0.5340 | 0.3604 | -17.36 | -32.5% | -9.43 | 0.001* |
| TSDAE-SAPT (reviews) + Query-DANN v2 | 0.5262 | 0.4860 | -4.03 | -7.7% | +3.90 | 0.001* |
| TSDAE-SAPT (task queries) | 0.5394 | 0.4555 | -8.39 | -15.6% | -0.46 | 0.001* |
| TSDAE-SAPT (task queries) + Query-DANN v2 | 0.5530 | 0.5205 | -3.25 | -5.9% | +4.69 | 0.001* |
