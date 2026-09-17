# Retrieval results — MIRACL-id test (578 queries, 500,000 passages)

BGE-M3 dense retrieval, top-100. Harmonised training protocol for every trained model.

| Model | Formal R@100 | Formal MRR@100 | Formal nDCG@10 | Informal R@100 | Informal MRR@100 | Informal nDCG@10 |
|---|---|---|---|---|---|---|
| BGE-M3 base | 0.9393 | 0.7282 | 0.6166 | 0.9003 | 0.6441 | 0.5373 |
| BGE-M3 + LLM normalisation | 0.9393 | 0.7282 | 0.6166 | 0.9369 | 0.7131 | 0.6017 |
| Query-DANN v2 (linear adapter) | 0.9429 | 0.7363 | 0.6234 | 0.9179 | 0.6864 | 0.5775 |
| Query distillation score (linear adapter) | 0.9422 | 0.7288 | 0.6175 | 0.9159 | 0.6875 | 0.5773 |
| TSDAE-SAPT (task queries) | 0.9399 | 0.7287 | 0.6182 | 0.8969 | 0.6373 | 0.5266 |
| TSDAE-SAPT (task queries) + Query-DANN v2 (linear adapter) | 0.9415 | 0.7347 | 0.6202 | 0.9186 | 0.6934 | 0.5789 |
| TSDAE-SAPT (task queries) + Query distillation score (linear adapter) | 0.9415 | 0.7214 | 0.6129 | 0.9176 | 0.6832 | 0.5735 |

## nDCG@10 formal/informal gap

Gap = informal − formal (negative = informal is worse). Narrowing vs base = how much of the base model's 7.93-point gap this pipeline removes. p tests informal vs formal within the model (paired bootstrap, 1000 samples).

| Model | Formal nDCG@10 | Informal nDCG@10 | Gap (pts) | Gap (%) | Narrowing vs base | p |
|---|---|---|---|---|---|---|
| BGE-M3 base | 0.6166 | 0.5373 | -7.93 | -12.9% | +0.00 | 0.001* |
| BGE-M3 + LLM normalisation | 0.6166 | 0.6017 | -1.49 | -2.4% | +6.44 | 0.002* |
| Query-DANN v2 (linear adapter) | 0.6234 | 0.5775 | -4.59 | -7.4% | +3.34 | 0.001* |
| Query distillation score (linear adapter) | 0.6175 | 0.5773 | -4.01 | -6.5% | +3.92 | 0.001* |
| TSDAE-SAPT (task queries) | 0.6182 | 0.5266 | -9.15 | -14.8% | -1.22 | 0.001* |
| TSDAE-SAPT (task queries) + Query-DANN v2 (linear adapter) | 0.6202 | 0.5789 | -4.13 | -6.7% | +3.80 | 0.001* |
| TSDAE-SAPT (task queries) + Query distillation score (linear adapter) | 0.6129 | 0.5735 | -3.94 | -6.4% | +4.00 | 0.001* |
