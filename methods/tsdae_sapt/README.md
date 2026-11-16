# TSDAE-SAPT (this folder)

Adapt BGE-M3 to informal/slang Indonesian by **unsupervised** continued pretraining
(TSDAE: delete ~60% of each review's words, encode to one `[CLS]` vector, and train a
decoder to reconstruct the original review). No labels, no adapters. **Status: negative
result — do not rerun.** See `TSDAE_SAPT_IMPLEMENTATION.md` for the results and why.

## Files

| File | Role |
|---|---|
| `train_sapt.py` | The whole TSDAE training: data → noise → encoder/decoder wiring → loop → save |
| `prepare_sapt_cache.py` | Re-encode queries (and optionally all 500k passages) with the trained encoder |
| `evaluate_sapt.py` | Zero-shot MIRACL-id eval of the 3 conditions + bootstrap + drift diagnostics |
| `outputs/`, `cache/` | Eval results + the saved 2.3 GB encoder with its `log.jsonl`; embedding caches |

## Which model layers are modified

| Component | Fate |
|---|---|
| BGE-M3 encoder (embeddings + 24 layers) | **Fully fine-tuned — this is the only thing saved** |
| Decoder self-attn / FFN / embeddings | Tied to encoder parameters by name, discarded after training |
| Decoder cross-attention + LM-head transform | New weights, discarded after training |
| Passage index (500k embeddings) | Valid only if passages are re-encoded (`symmetric` mode) |

## Run

```bash
bash models/tsdae_sapt/run.sh   # train (~5 min) → caches → eval on test + dev
```
