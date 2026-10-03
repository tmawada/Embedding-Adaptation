# Mean ± SD over training seeds — MIRACL-id test (578 queries, 500,000 passages)

BGE-M3 dense retrieval. Trained models: mean ± sample SD over seeds [42, 43, 44, 45, 46] (n per row in the JSON; harmonized protocol, gamma = 1.0, kd_weight = 1.0). The TSDAE (task queries) tower is fixed at its seed-42 checkpoint; only the adapters vary. BGE-M3 base and LLM normalisation are deterministic (single run, no SD).

| Model | Formal R@100 | Formal MRR@100 | Formal nDCG@10 | Informal R@100 | Informal MRR@100 | Informal nDCG@10 |
|---|---|---|---|---|---|---|
| BGE-M3 base | 0.9393 | 0.7282 | 0.6166 | 0.9003 | 0.6441 | 0.5373 |
| BGE-M3 + LLM normalisation | 0.9393 | 0.7282 | 0.6166 | 0.9369 | 0.7131 | 0.6017 |
| Query-DANN v2 (basic adapter) | 0.9403 ± 0.0025 | 0.7302 ± 0.0047 | 0.6173 ± 0.0029 | 0.9141 ± 0.0013 | 0.6873 ± 0.0036 | 0.5763 ± 0.0029 |
| Query-DANN v2 (linear adapter) | 0.9426 ± 0.0004 | 0.7362 ± 0.0014 | 0.6228 ± 0.0008 | 0.9179 ± 0.0004 | 0.6864 ± 0.0007 | 0.5775 ± 0.0008 |
| Query distillation score (basic adapter) | 0.9422 ± 0.0009 | 0.7260 ± 0.0020 | 0.6169 ± 0.0014 | 0.9149 ± 0.0021 | 0.6826 ± 0.0007 | 0.5755 ± 0.0017 |
| Query distillation score (linear adapter) | 0.9422 ± 0.0002 | 0.7279 ± 0.0014 | 0.6172 ± 0.0012 | 0.9159 ± 0.0013 | 0.6862 ± 0.0011 | 0.5765 ± 0.0018 |
| Query distillation embed (basic adapter) | 0.9413 ± 0.0007 | 0.7313 ± 0.0024 | 0.6178 ± 0.0008 | 0.9102 ± 0.0010 | 0.6779 ± 0.0021 | 0.5648 ± 0.0018 |
| Query distillation embed (linear adapter) | 0.9400 ± 0.0007 | 0.7333 ± 0.0009 | 0.6181 ± 0.0004 | 0.9098 ± 0.0010 | 0.6817 ± 0.0030 | 0.5658 ± 0.0022 |
| TSDAE-SAPT (task queries) + Query-DANN v2 (linear adapter) | 0.9430 ± 0.0011 | 0.7340 ± 0.0012 | 0.6205 ± 0.0012 | 0.9188 ± 0.0008 | 0.6904 ± 0.0025 | 0.5774 ± 0.0018 |
| TSDAE-SAPT (task queries) + Query distillation score (linear adapter) | 0.9410 ± 0.0009 | 0.7216 ± 0.0013 | 0.6129 ± 0.0008 | 0.9173 ± 0.0005 | 0.6843 ± 0.0014 | 0.5737 ± 0.0005 |
| TSDAE-SAPT (task queries) + Query-DANN v2 + distillation score (linear adapter) | 0.9419 ± 0.0009 | 0.7325 ± 0.0034 | 0.6197 ± 0.0014 | 0.9186 ± 0.0010 | 0.6882 ± 0.0014 | 0.5766 ± 0.0008 |

## nDCG@10 formal/informal gap (mean ± SD over seeds)

Gap = informal − formal in nDCG@10 points. Narrowing vs base = gap − base gap (-7.93).

| Model | Formal nDCG@10 | Informal nDCG@10 | Gap (pts) | Narrowing vs base (pts) |
|---|---|---|---|---|
| BGE-M3 base | 0.6166 | 0.5373 | -7.93 | +0.00 |
| BGE-M3 + LLM normalisation | 0.6166 | 0.6017 | -1.49 | +6.44 |
| Query-DANN v2 (basic adapter) | 0.6173 ± 0.0029 | 0.5763 ± 0.0029 | -4.10 ± 0.13 | +3.83 ± 0.13 |
| Query-DANN v2 (linear adapter) | 0.6228 ± 0.0008 | 0.5775 ± 0.0008 | -4.54 ± 0.13 | +3.40 ± 0.13 |
| Query distillation score (basic adapter) | 0.6169 ± 0.0014 | 0.5755 ± 0.0017 | -4.14 ± 0.27 | +3.79 ± 0.27 |
| Query distillation score (linear adapter) | 0.6172 ± 0.0012 | 0.5765 ± 0.0018 | -4.07 ± 0.06 | +3.86 ± 0.06 |
| Query distillation embed (basic adapter) | 0.6178 ± 0.0008 | 0.5648 ± 0.0018 | -5.29 ± 0.16 | +2.64 ± 0.16 |
| Query distillation embed (linear adapter) | 0.6181 ± 0.0004 | 0.5658 ± 0.0022 | -5.22 ± 0.22 | +2.71 ± 0.22 |
| TSDAE-SAPT (task queries) + Query-DANN v2 (linear adapter) | 0.6205 ± 0.0012 | 0.5774 ± 0.0018 | -4.31 ± 0.15 | +3.62 ± 0.15 |
| TSDAE-SAPT (task queries) + Query distillation score (linear adapter) | 0.6129 ± 0.0008 | 0.5737 ± 0.0005 | -3.93 ± 0.10 | +4.00 ± 0.10 |
| TSDAE-SAPT (task queries) + Query-DANN v2 + distillation score (linear adapter) | 0.6197 ± 0.0014 | 0.5766 ± 0.0008 | -4.30 ± 0.22 | +3.63 ± 0.22 |

## Key comparisons (nDCG@10 points)

Across seeds = per-seed difference, mean ± SD. p = paired bootstrap on seed-averaged per-query scores.

| A | B | Informal A − B (seeds) | p | Formal A − B (seeds) | p |
|---|---|---|---|---|---|
| Query-DANN v2 (linear adapter) | Query-DANN v2 (basic adapter) | +0.11 ± 0.27 | 0.685 | +0.55 ± 0.35 | 0.035* |
| Query-DANN v2 (linear adapter) | Query distillation score (linear adapter) | +0.09 ± 0.12 | 0.790 | +0.56 ± 0.16 | 0.045* |
| Query distillation score (linear adapter) | Query distillation embed (linear adapter) | +1.07 ± 0.28 | 0.003* | -0.08 ± 0.10 | 0.708 |
| TSDAE-SAPT (task queries) + Query-DANN v2 (linear adapter) | Query-DANN v2 (linear adapter) | -0.01 ± 0.21 | 0.981 | -0.24 ± 0.14 | 0.160 |
| TSDAE-SAPT (task queries) + Query-DANN v2 + distillation score (linear adapter) | Query-DANN v2 (linear adapter) | -0.08 ± 0.13 | 0.704 | -0.31 ± 0.20 | 0.125 |
