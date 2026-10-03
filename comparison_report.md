# Model Comparison Report - TEST Split

Generated: 2026-10-03 13:42:43

## Overview

Comprehensive comparison of 9 models on MIRACL Indonesian retrieval:
- **Base Models**: Query DANN and Query Distillation with Linear/Adapter variants
- **TSDAE Models**: TSDAE pre-training + Query DANN/Distillation
- **Metrics**: Recall@100, MRR@10, nDCG@10
- **Main Focus**: Formal vs Informal gap (nDCG@10)

## Results Table

| Model | Informal nDCG@10 | Formal nDCG@10 | Gap (F-I) | Informal MRR@10 | Formal MRR@10 | Informal R@100 | Formal R@100 |
|-------|------------------|----------------|-----------|-----------------|---------------|----------------|--------------|
| Baseline (BGE-M3)         | 0.5373 | 0.6166 | +0.0793 | 0.6391 | 0.7248 | 0.9003 | 0.9393 |
| Query DANN (Linear)       | 0.5584 | 0.6149 | +0.0564 | 0.6664 | 0.7182 | 0.9104 | 0.9394 |
| Query DANN (Adapter)      | 0.5540 | 0.6131 | +0.0591 | 0.6601 | 0.7221 | 0.8997 | 0.9415 |
| Query Dist (Linear)       | 0.5773 | 0.6175 | +0.0401 | 0.6832 | 0.7256 | 0.9159 | 0.9422 |
| Query Dist (Adapter)      | 0.5730 | 0.6177 | +0.0447 | 0.6785 | 0.7230 | 0.9164 | 0.9418 |
| TSDAE Only                | 0.2468 | 0.4289 | +0.1820 | 0.3073 | 0.5283 | 0.6420 | 0.8223 |
| TSDAE + DANN (Linear)     | 0.4136 | 0.4840 | +0.0704 | 0.5069 | 0.5875 | 0.7905 | 0.8753 |
| TSDAE + DANN (Adapter)    | 0.4139 | 0.4629 | +0.0490 | 0.5068 | 0.5667 | 0.7838 | 0.8465 |
| TSDAE + Dist (Linear)     | 0.3585 | 0.4479 | +0.0894 | 0.4408 | 0.5464 | 0.7467 | 0.8402 |
| TSDAE + Dist (Adapter)    | 0.3548 | 0.4360 | +0.0812 | 0.4357 | 0.5350 | 0.7408 | 0.8244 |

## Gap Analysis (nDCG@10)

Positive gap = Formal queries outperform Informal queries

### Gap Reduction from Baseline

**Baseline Gap**: +0.0793

- **Query DANN (Linear)**: Gap=+0.0564, Reduction=+0.0229 (+28.8%)
- **Query DANN (Adapter)**: Gap=+0.0591, Reduction=+0.0202 (+25.5%)
- **Query Dist (Linear)**: Gap=+0.0401, Reduction=+0.0392 (+49.4%)
- **Query Dist (Adapter)**: Gap=+0.0447, Reduction=+0.0346 (+43.7%)
- **TSDAE Only**: Gap=+0.1820, Reduction=-0.1027 (-129.5%)
- **TSDAE + DANN (Linear)**: Gap=+0.0704, Reduction=+0.0089 (+11.3%)
- **TSDAE + DANN (Adapter)**: Gap=+0.0490, Reduction=+0.0303 (+38.2%)
- **TSDAE + Dist (Linear)**: Gap=+0.0894, Reduction=-0.0101 (-12.7%)
- **TSDAE + Dist (Adapter)**: Gap=+0.0812, Reduction=-0.0019 (-2.3%)

## Statistical Significance

Paired bootstrap tests (1000 samples) for gap differences:

### Baseline (BGE-M3)
- Gap: +0.0793
- 95% CI: [+0.0634, +0.0960]
- p-value: 0.0010

### Query DANN (Linear)
- Gap: +0.0564
- 95% CI: [+0.0436, +0.0711]
- p-value: 0.0010

### Query DANN (Adapter)
- Gap: +0.0591
- 95% CI: [+0.0442, +0.0743]
- p-value: 0.0010

### Query Dist (Linear)
- Gap: +0.0401
- 95% CI: [+0.0280, +0.0539]
- p-value: 0.0010

### Query Dist (Adapter)
- Gap: +0.0447
- 95% CI: [+0.0320, +0.0582]
- p-value: 0.0010

### TSDAE Only
- Gap: +0.1820
- 95% CI: [+0.1600, +0.2052]
- p-value: 0.0010

### TSDAE + DANN (Linear)
- Gap: +0.0704
- 95% CI: [+0.0504, +0.0896]
- p-value: 0.0010

### TSDAE + DANN (Adapter)
- Gap: +0.0490
- 95% CI: [+0.0282, +0.0690]
- p-value: 0.0010

### TSDAE + Dist (Linear)
- Gap: +0.0894
- 95% CI: [+0.0722, +0.1076]
- p-value: 0.0010

### TSDAE + Dist (Adapter)
- Gap: +0.0812
- 95% CI: [+0.0635, +0.0983]
- p-value: 0.0010


## Best Performers

- **Best Informal nDCG@10**: dist_linear (0.5773)
- **Best Formal nDCG@10**: dist_adapter (0.6177)
- **Smallest Gap**: dist_linear (+0.0401)
