# Informal-query retrieval on MIRACL-id test (578 queries, 500,000 passages)

All rows: BGE-M3 **dense** retrieval. Gap closed = share of the 7.93-point formal/informal nDCG@10 gap recovered. Δ = nDCG@10 points vs zero-shot; * = p < 0.05 (paired bootstrap, 1000 samples).

| System | nDCG@10 | MRR@10 | R@100 | Gap closed | Δ vs zero-shot |
|---|---|---|---|---|---|
| BGE-M3 base - formal query (upper bound) | 0.6166 | 0.7248 | 0.9393 |  |  |
| BGE-M3 base - informal query (zero-shot) | 0.5373 | 0.6391 | 0.9003 |  |  |
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | 0.6017 | 0.7101 | 0.9369 | 81.2% | +6.44* (p=0.001) |
| Query-DANN v2 (adapter on base tower) | 0.5748 | 0.6847 | 0.9084 | 47.3% | +3.75* (p=0.001) |
| Query distillation: score (adapter on base tower) | 0.5768 | 0.6826 | 0.9142 | 49.8% | +3.95* (p=0.001) |
| Query distillation: embed (adapter on base tower) | 0.5672 | 0.6770 | 0.9135 | 37.7% | +2.99* (p=0.001) |
| SAPT-reviews tower - informal query (no adapter) | 0.3604 | 0.4423 | 0.7524 | -223.1% | -17.70* (p=0.001) |
| SAPT-reviews tower - formal query | 0.5340 | 0.6406 | 0.8942 |  | -0.33 (p=0.762) |
| SAPT-reviews + Query-DANN v2 (full stack) | 0.4860 | 0.5967 | 0.8241 | -64.7% | -5.13* (p=0.001) |
| SAPT-reviews + distillation: score | 0.4732 | 0.5815 | 0.8239 | -80.8% | -6.41* (p=0.001) |
| SAPT-queries tower - informal query (no adapter) | 0.4555 | 0.5476 | 0.8390 | -103.2% | -8.18* (p=0.001) |
| SAPT-queries tower - formal query | 0.5394 | 0.6437 | 0.8958 |  | +0.21 (p=0.857) |
| SAPT-queries + Query-DANN v2 (full stack) | 0.5205 | 0.6249 | 0.8613 | -21.1% | -1.68 (p=0.115) |
| SAPT-queries + distillation: score | 0.4985 | 0.6019 | 0.8523 | -48.9% | -3.88* (p=0.001) |
| SAPT-reviews symmetric (SAPT queries AND passages) | 0.2468 | 0.3073 | 0.6420 | -366.3% | -29.05* (p=0.001) |

## Head-to-head (nDCG@10)

| A | B | A - B | p |
|---|---|---|---|
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | Query-DANN v2 (adapter on base tower) | +2.69* | 0.002 |
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | Query distillation: score (adapter on base tower) | +2.49* | 0.001 |
| BGE-M3 + LLM normalisation (gemini-3.5-flash-lite) | Query distillation: embed (adapter on base tower) | +3.46* | 0.001 |
| SAPT-reviews tower - informal query (no adapter) | Query-DANN v2 (adapter on base tower) | -21.44* | 0.001 |
| SAPT-reviews tower - informal query (no adapter) | Query distillation: score (adapter on base tower) | -21.65* | 0.001 |
| SAPT-reviews tower - informal query (no adapter) | Query distillation: embed (adapter on base tower) | -20.68* | 0.001 |
| SAPT-reviews + Query-DANN v2 (full stack) | Query-DANN v2 (adapter on base tower) | -8.88* | 0.001 |
| SAPT-reviews + Query-DANN v2 (full stack) | Query distillation: score (adapter on base tower) | -9.09* | 0.001 |
| SAPT-reviews + Query-DANN v2 (full stack) | Query distillation: embed (adapter on base tower) | -8.12* | 0.001 |
| SAPT-reviews + distillation: score | Query-DANN v2 (adapter on base tower) | -10.16* | 0.001 |
| SAPT-reviews + distillation: score | Query distillation: score (adapter on base tower) | -10.36* | 0.001 |
| SAPT-reviews + distillation: score | Query distillation: embed (adapter on base tower) | -9.39* | 0.001 |
| SAPT-queries tower - informal query (no adapter) | Query-DANN v2 (adapter on base tower) | -11.93* | 0.001 |
| SAPT-queries tower - informal query (no adapter) | Query distillation: score (adapter on base tower) | -12.14* | 0.001 |
| SAPT-queries tower - informal query (no adapter) | Query distillation: embed (adapter on base tower) | -11.17* | 0.001 |
| SAPT-queries + Query-DANN v2 (full stack) | Query-DANN v2 (adapter on base tower) | -5.43* | 0.001 |
| SAPT-queries + Query-DANN v2 (full stack) | Query distillation: score (adapter on base tower) | -5.63* | 0.001 |
| SAPT-queries + Query-DANN v2 (full stack) | Query distillation: embed (adapter on base tower) | -4.66* | 0.001 |
| SAPT-queries + distillation: score | Query-DANN v2 (adapter on base tower) | -7.63* | 0.001 |
| SAPT-queries + distillation: score | Query distillation: score (adapter on base tower) | -7.83* | 0.001 |
| SAPT-queries + distillation: score | Query distillation: embed (adapter on base tower) | -6.86* | 0.001 |
| SAPT-reviews symmetric (SAPT queries AND passages) | Query-DANN v2 (adapter on base tower) | -32.80* | 0.001 |
| SAPT-reviews symmetric (SAPT queries AND passages) | Query distillation: score (adapter on base tower) | -33.00* | 0.001 |
| SAPT-reviews symmetric (SAPT queries AND passages) | Query distillation: embed (adapter on base tower) | -32.04* | 0.001 |
| SAPT-reviews + Query-DANN v2 (full stack) | SAPT-reviews tower - informal query (no adapter) | +12.56* | 0.001 |
| SAPT-queries + Query-DANN v2 (full stack) | SAPT-queries tower - informal query (no adapter) | +6.51* | 0.001 |
