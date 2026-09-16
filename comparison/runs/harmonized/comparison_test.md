# Informal-query retrieval on MIRACL-id test (578 queries, 500,000 passages)

All rows: BGE-M3 **dense** retrieval. Gap closed = share of the 7.93-point formal/informal nDCG@10 gap recovered. Δ = nDCG@10 points vs zero-shot; * = p < 0.05 (paired bootstrap, 1000 samples).

| System | nDCG@10 | MRR@10 | R@100 | Gap closed | Δ vs zero-shot |
|---|---|---|---|---|---|
| BGE-M3 base - formal query (upper bound) | 0.6166 | 0.7248 | 0.9393 |  |  |
| BGE-M3 base - informal query (zero-shot) | 0.5373 | 0.6391 | 0.9003 |  |  |
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | 0.6017 | 0.7101 | 0.9369 | 81.2% | +6.44* (p=0.001) |
| Query-DANN v2 (adapter on base tower) | 0.5738 | 0.6783 | 0.9145 | 46.0% | +3.65* (p=0.001) |
| Query distillation: score (adapter on base tower) | 0.5730 | 0.6785 | 0.9164 | 45.0% | +3.57* (p=0.001) |
| Query distillation: embed (adapter on base tower) | 0.5652 | 0.6751 | 0.9118 | 35.2% | +2.79* (p=0.001) |
| SAPT-reviews tower - informal query (no adapter) | 0.5351 | 0.6377 | 0.8965 | -2.7% | -0.22 (p=0.433) |
| SAPT-reviews tower - formal query | 0.6143 | 0.7188 | 0.9390 |  | +7.70* (p=0.001) |
| SAPT-reviews + Query-DANN v2 (full stack) | 0.5758 | 0.6831 | 0.9110 | 48.5% | +3.85* (p=0.001) |
| SAPT-reviews + distillation: score | 0.5693 | 0.6768 | 0.9099 | 40.3% | +3.20* (p=0.001) |
| SAPT-queries tower - informal query (no adapter) | 0.5266 | 0.6322 | 0.8969 | -13.5% | -1.07* (p=0.001) |
| SAPT-queries tower - formal query | 0.6182 | 0.7254 | 0.9399 |  | +8.08* (p=0.001) |
| SAPT-queries + Query-DANN v2 (full stack) | 0.5754 | 0.6785 | 0.9154 | 48.1% | +3.81* (p=0.001) |
| SAPT-queries + distillation: score | 0.5727 | 0.6776 | 0.9154 | 44.6% | +3.53* (p=0.001) |

## Head-to-head (nDCG@10)

| A | B | A - B | p |
|---|---|---|---|
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | Query-DANN v2 (adapter on base tower) | +2.79* | 0.001 |
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | Query distillation: score (adapter on base tower) | +2.87* | 0.001 |
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | Query distillation: embed (adapter on base tower) | +3.65* | 0.001 |
| SAPT-reviews tower - informal query (no adapter) | Query-DANN v2 (adapter on base tower) | -3.87* | 0.001 |
| SAPT-reviews tower - informal query (no adapter) | Query distillation: score (adapter on base tower) | -3.79* | 0.001 |
| SAPT-reviews tower - informal query (no adapter) | Query distillation: embed (adapter on base tower) | -3.01* | 0.001 |
| SAPT-reviews + Query-DANN v2 (full stack) | Query-DANN v2 (adapter on base tower) | +0.20 | 0.369 |
| SAPT-reviews + Query-DANN v2 (full stack) | Query distillation: score (adapter on base tower) | +0.28 | 0.542 |
| SAPT-reviews + Query-DANN v2 (full stack) | Query distillation: embed (adapter on base tower) | +1.05 | 0.055 |
| SAPT-reviews + distillation: score | Query-DANN v2 (adapter on base tower) | -0.45 | 0.394 |
| SAPT-reviews + distillation: score | Query distillation: score (adapter on base tower) | -0.37 | 0.169 |
| SAPT-reviews + distillation: score | Query distillation: embed (adapter on base tower) | +0.41 | 0.341 |
| SAPT-queries tower - informal query (no adapter) | Query-DANN v2 (adapter on base tower) | -4.72* | 0.001 |
| SAPT-queries tower - informal query (no adapter) | Query distillation: score (adapter on base tower) | -4.64* | 0.001 |
| SAPT-queries tower - informal query (no adapter) | Query distillation: embed (adapter on base tower) | -3.86* | 0.001 |
| SAPT-queries + Query-DANN v2 (full stack) | Query-DANN v2 (adapter on base tower) | +0.16 | 0.500 |
| SAPT-queries + Query-DANN v2 (full stack) | Query distillation: score (adapter on base tower) | +0.24 | 0.604 |
| SAPT-queries + Query-DANN v2 (full stack) | Query distillation: embed (adapter on base tower) | +1.02 | 0.078 |
| SAPT-queries + distillation: score | Query-DANN v2 (adapter on base tower) | -0.11 | 0.816 |
| SAPT-queries + distillation: score | Query distillation: score (adapter on base tower) | -0.04 | 0.912 |
| SAPT-queries + distillation: score | Query distillation: embed (adapter on base tower) | +0.74 | 0.079 |
| SAPT-reviews + Query-DANN v2 (full stack) | SAPT-reviews tower - informal query (no adapter) | +4.06* | 0.001 |
| SAPT-queries + Query-DANN v2 (full stack) | SAPT-queries tower - informal query (no adapter) | +4.88* | 0.001 |
