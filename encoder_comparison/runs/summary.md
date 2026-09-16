# Embedding model comparison — MIRACL-id test split

578 held-out queries, 500,000 passages. Δ values are nDCG@10 points; * = p < 0.05 (paired bootstrap, 1000 samples). Identical splits, methods, hyperparameters and step budgets for both encoders.

## Zero-shot robustness

| Encoder | Formal nDCG@10 | Informal nDCG@10 | Drop (pts) | Drop % | Formal MRR@10 | Informal MRR@10 | Formal R@100 | Informal R@100 |
|---|---|---|---|---|---|---|---|---|
| BGE-M3 | 0.6166 | 0.5373 | -7.93* | -12.9% | 0.7248 | 0.6391 | 0.9393 | 0.9003 |
| mE5-large-instruct | 0.5558 | 0.5142 | -4.16* | -7.5% | 0.6784 | 0.6323 | 0.9070 | 0.8699 |

- mE5-large-instruct - BGE-M3 (formal): -6.08* nDCG@10 (p=0.001)

- mE5-large-instruct - BGE-M3 (informal): -2.31* nDCG@10 (p=0.026)

- mE5-large-instruct - BGE-M3 (informal drop): +3.77* nDCG@10 (p=0.001)

Cosine-score scale (formal queries): BGE-M3 positive 0.621 / top-10 non-relevant 0.604 / random 0.259; mE5-large-instruct positive 0.890 / top-10 non-relevant 0.887 / random 0.757

## Methods on informal queries: nDCG@10

| Method | BGE-M3 nDCG@10 | gap closed | Δ vs zero-shot | formal Δ | mE5-large-instruct nDCG@10 | gap closed | Δ vs zero-shot | formal Δ |
|---|---|---|---|---|---|---|---|---|
| Query-DANN v2 (γ picked on dev) | 0.5748 | 47.3% | +3.75* | -0.45 | 0.5181 | 9.4% | +0.39 | -1.41* |
| Adapter + InfoNCE only (γ=0) | 0.5509 | 17.1% | +1.36 | -0.05 | 0.4995 | -35.5% | -1.48 | -1.51* |
| Distillation: embed | 0.5672 | 37.7% | +2.99* | +0.27 | 0.5289 | 35.2% | +1.46* | +0.06 |
| Distillation: score | 0.5768 | 49.8% | +3.95* | +0.37 | 0.5269 | 30.4% | +1.26* | -0.34 |
| Distillation: embed + InfoNCE | 0.5725 | 44.4% | +3.52* | +0.64 | 0.5186 | 10.4% | +0.43 | -1.00* |
| Distillation: score + InfoNCE | 0.5716 | 43.2% | +3.43* | +0.55 | 0.5135 | -1.9% | -0.08 | -1.22* |
| UDA disjoint: source-only (γ=0) | 0.5143 | -29.0% | -2.30* | -1.83* | 0.5040 | -24.6% | -1.02 | -1.66* |
| UDA disjoint: γ=0.1 | 0.5474 | 12.7% | +1.01 | -1.10* | 0.5037 | -25.3% | -1.05 | -2.68* |
| UDA disjoint: γ=0.5 | 0.5497 | 15.6% | +1.24 | -1.56* | 0.4999 | -34.4% | -1.43* | -2.60* |
| UDA disjoint: γ=1.0 | 0.5417 | 5.5% | +0.44 | -1.15 | 0.5085 | -13.8% | -0.57 | -2.40* |
| UDA overlap: source-only (γ=0) | 0.5273 | -12.7% | -1.00 | -0.74 | 0.5127 | -3.6% | -0.15 | -1.77* |
| UDA overlap: γ=1.0 | 0.5580 | 26.1% | +2.07* | -0.92 | 0.5105 | -9.0% | -0.37 | -1.07 |

## Methods on informal queries: MRR@10 and Recall@100

| Method | BGE-M3 MRR@10 | BGE-M3 R@100 | mE5-large-instruct MRR@10 | mE5-large-instruct R@100 |
|---|---|---|---|---|
| Query-DANN v2 (γ picked on dev) | 0.6847 | 0.9084 | 0.6345 | 0.8794 |
| Adapter + InfoNCE only (γ=0) | 0.6626 | 0.8995 | 0.6062 | 0.8586 |
| Distillation: embed | 0.6770 | 0.9135 | 0.6459 | 0.8766 |
| Distillation: score | 0.6826 | 0.9142 | 0.6409 | 0.8821 |
| Distillation: embed + InfoNCE | 0.6801 | 0.9132 | 0.6393 | 0.8704 |
| Distillation: score + InfoNCE | 0.6799 | 0.9053 | 0.6242 | 0.8627 |
| UDA disjoint: source-only (γ=0) | 0.6211 | 0.8790 | 0.6261 | 0.8623 |
| UDA disjoint: γ=0.1 | 0.6508 | 0.8903 | 0.6223 | 0.8544 |
| UDA disjoint: γ=0.5 | 0.6546 | 0.8991 | 0.6195 | 0.8520 |
| UDA disjoint: γ=1.0 | 0.6545 | 0.8851 | 0.6292 | 0.8570 |
| UDA overlap: source-only (γ=0) | 0.6334 | 0.8795 | 0.6207 | 0.8594 |
| UDA overlap: γ=1.0 | 0.6653 | 0.9034 | 0.6287 | 0.8559 |

## Selected Query-DANN v2 checkpoints (γ picked on dev nDCG@10)

- BGE-M3: `runs/v2_dann_g1.0/best`
- mE5-large-instruct: `encoder_comparison/runs/me5_large_instruct/query_dann_v2/g1.0/best`
