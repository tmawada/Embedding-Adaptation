# Informal-query retrieval on MIRACL-id test (578 queries, 500,000 passages)

All rows: BGE-M3 **dense** retrieval over the same frozen passage index.
Gap closed = fraction of the 7.93-point formal/informal nDCG@10 gap recovered.
Δ = change vs zero-shot in nDCG@10 points; * = p < 0.05 (paired bootstrap, 1000 samples).

| System | nDCG@10 | MRR@10 | R@100 | Gap closed | Δ vs zero-shot |
|---|---|---|---|---|---|
| BGE-M3 base | formal query (upper bound) | 0.6166 | 0.7248 | 0.9393 |  |  |
| BGE-M3 base | informal query (zero-shot) | 0.5373 | 0.6391 | 0.9003 |  |  |
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | 0.6017 | 0.7101 | 0.9369 | 81.2% | +6.44* (p=0.001) |
| Query-DANN v2 (adapter, adversarial) | 0.5748 | 0.6847 | 0.9084 | 47.3% | +3.75* (p=0.001) |
| Query distillation: score (adapter) | 0.5768 | 0.6826 | 0.9142 | 49.8% | +3.95* (p=0.001) |
| Query distillation: embed (adapter) | 0.5672 | 0.6770 | 0.9135 | 37.7% | +2.99* (p=0.001) |
| TSDAE-DAPT tower | informal query (no adapter) | 0.4555 | 0.5476 | 0.8390 | -103.2% | -8.18* (p=0.001) |
| TSDAE-DAPT tower | formal query | 0.5394 | 0.6437 | 0.8958 |  | +0.21 (p=0.857) |
| TSDAE-DAPT + Query-DANN v2 (full stack) | 0.5205 | 0.6249 | 0.8613 | -21.1% | -1.68 (p=0.115) |
| TSDAE-DAPT + distillation: score | 0.4985 | 0.6019 | 0.8523 | -48.9% | -3.88* (p=0.001) |

## Head-to-head (nDCG@10)

| A | B | A − B | p |
|---|---|---|---|
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | Query-DANN v2 (adapter, adversarial) | +2.69* | 0.002 |
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | Query distillation: score (adapter) | +2.49* | 0.001 |
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | Query distillation: embed (adapter) | +3.46* | 0.001 |
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | TSDAE-DAPT tower | informal query (no adapter) | +14.62* | 0.001 |
| TSDAE-DAPT + Query-DANN v2 (full stack) | Query-DANN v2 (adapter, adversarial) | -5.43* | 0.001 |
| TSDAE-DAPT + Query-DANN v2 (full stack) | Query distillation: score (adapter) | -5.63* | 0.001 |
| TSDAE-DAPT + Query-DANN v2 (full stack) | Query distillation: embed (adapter) | -4.66* | 0.001 |
| TSDAE-DAPT + Query-DANN v2 (full stack) | TSDAE-DAPT tower | informal query (no adapter) | +6.51* | 0.001 |
| TSDAE-DAPT + distillation: score | Query-DANN v2 (adapter, adversarial) | -7.63* | 0.001 |
| TSDAE-DAPT + distillation: score | Query distillation: score (adapter) | -7.83* | 0.001 |
| TSDAE-DAPT + distillation: score | Query distillation: embed (adapter) | -6.86* | 0.001 |
| TSDAE-DAPT + distillation: score | TSDAE-DAPT tower | informal query (no adapter) | +4.31* | 0.001 |
