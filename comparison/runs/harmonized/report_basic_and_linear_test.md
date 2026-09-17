# Retrieval results — MIRACL-id test (578 queries, 500,000 passages)

BGE-M3 dense retrieval, top-100. Harmonised training protocol for every trained model.

| Model | Formal R@100 | Formal MRR@100 | Formal nDCG@10 | Informal R@100 | Informal MRR@100 | Informal nDCG@10 |
|---|---|---|---|---|---|---|
| BGE-M3 base | 0.9393 | 0.7282 | 0.6166 | 0.9003 | 0.6441 | 0.5373 |
| BGE-M3 + LLM normalisation | 0.9393 | 0.7282 | 0.6166 | 0.9369 | 0.7131 | 0.6017 |
| Query-DANN v2 (basic adapter) | 0.9384 | 0.7277 | 0.6136 | 0.9145 | 0.6831 | 0.5738 |
| Query-DANN v2 (linear adapter) | 0.9429 | 0.7363 | 0.6234 | 0.9179 | 0.6864 | 0.5775 |
| Query distillation score (basic adapter) | 0.9418 | 0.7261 | 0.6177 | 0.9164 | 0.6831 | 0.5730 |
| Query distillation score (linear adapter) | 0.9422 | 0.7288 | 0.6175 | 0.9159 | 0.6875 | 0.5773 |
| Query distillation embed (basic adapter) | 0.9407 | 0.7330 | 0.6185 | 0.9118 | 0.6797 | 0.5652 |
| Query distillation embed (linear adapter) | 0.9399 | 0.7336 | 0.6186 | 0.9085 | 0.6821 | 0.5661 |
| TSDAE-SAPT (task queries) | 0.9399 | 0.7287 | 0.6182 | 0.8969 | 0.6373 | 0.5266 |
| TSDAE-SAPT (task queries) + Query-DANN v2 (basic adapter) | 0.9391 | 0.7233 | 0.6113 | 0.9154 | 0.6827 | 0.5754 |
| TSDAE-SAPT (task queries) + Query-DANN v2 (linear adapter) | 0.9415 | 0.7347 | 0.6202 | 0.9186 | 0.6934 | 0.5789 |
| TSDAE-SAPT (task queries) + Query distillation score (basic adapter) | 0.9412 | 0.7249 | 0.6162 | 0.9154 | 0.6820 | 0.5727 |
| TSDAE-SAPT (task queries) + Query distillation score (linear adapter) | 0.9415 | 0.7214 | 0.6129 | 0.9176 | 0.6832 | 0.5735 |
| TSDAE-SAPT (task queries) + Query distillation embed (basic adapter) | 0.9380 | 0.7259 | 0.6164 | 0.9083 | 0.6767 | 0.5634 |
| TSDAE-SAPT (task queries) + Query distillation embed (linear adapter) | 0.9382 | 0.7207 | 0.6137 | 0.9093 | 0.6770 | 0.5635 |

## nDCG@10 formal/informal gap

Gap = informal − formal (negative = informal is worse). Narrowing vs base = how much of the base model's 7.93-point gap this pipeline removes. p tests informal vs formal within the model (paired bootstrap, 1000 samples).

| Model | Formal nDCG@10 | Informal nDCG@10 | Gap (pts) | Gap (%) | Narrowing vs base | p |
|---|---|---|---|---|---|---|
| BGE-M3 base | 0.6166 | 0.5373 | -7.93 | -12.9% | +0.00 | 0.001* |
| BGE-M3 + LLM normalisation | 0.6166 | 0.6017 | -1.49 | -2.4% | +6.44 | 0.002* |
| Query-DANN v2 (basic adapter) | 0.6136 | 0.5738 | -3.98 | -6.5% | +3.95 | 0.001* |
| Query-DANN v2 (linear adapter) | 0.6234 | 0.5775 | -4.59 | -7.4% | +3.34 | 0.001* |
| Query distillation score (basic adapter) | 0.6177 | 0.5730 | -4.47 | -7.2% | +3.46 | 0.001* |
| Query distillation score (linear adapter) | 0.6175 | 0.5773 | -4.01 | -6.5% | +3.92 | 0.001* |
| Query distillation embed (basic adapter) | 0.6185 | 0.5652 | -5.32 | -8.6% | +2.61 | 0.001* |
| Query distillation embed (linear adapter) | 0.6186 | 0.5661 | -5.24 | -8.5% | +2.69 | 0.001* |
| TSDAE-SAPT (task queries) | 0.6182 | 0.5266 | -9.15 | -14.8% | -1.22 | 0.001* |
| TSDAE-SAPT (task queries) + Query-DANN v2 (basic adapter) | 0.6113 | 0.5754 | -3.59 | -5.9% | +4.35 | 0.001* |
| TSDAE-SAPT (task queries) + Query-DANN v2 (linear adapter) | 0.6202 | 0.5789 | -4.13 | -6.7% | +3.80 | 0.001* |
| TSDAE-SAPT (task queries) + Query distillation score (basic adapter) | 0.6162 | 0.5727 | -4.36 | -7.1% | +3.57 | 0.001* |
| TSDAE-SAPT (task queries) + Query distillation score (linear adapter) | 0.6129 | 0.5735 | -3.94 | -6.4% | +4.00 | 0.001* |
| TSDAE-SAPT (task queries) + Query distillation embed (basic adapter) | 0.6164 | 0.5634 | -5.30 | -8.6% | +2.63 | 0.001* |
| TSDAE-SAPT (task queries) + Query distillation embed (linear adapter) | 0.6137 | 0.5635 | -5.02 | -8.2% | +2.91 | 0.001* |
