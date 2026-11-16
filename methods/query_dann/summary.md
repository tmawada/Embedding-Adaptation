# Query-DANN Adaptation Pipeline Summary

## Overview
Query-DANN adapts a frozen **BGE-M3** dense encoder for Indonesian RAG by learning a lightweight query-side adapter that maps informal/noisy queries into the same vector space as formal queries and documents — eliminating the need for an LLM rewrite at inference time.

---

## 1. Training Pipeline (`train.py`)

### Architecture
```
[Informal Query] ──► [BGE-M3 (frozen)] ──► [Adapter A_ψ] ──► z_q ──┬──► InfoNCE Loss
                                                                   │
[Formal Query/Doc] ──► [BGE-M3 (frozen)] ───────────────────────► z_f ──┤
                                                                   │
                                                                   └──► GRL(α) ──► [Discriminator D_φ] ──► BCE Loss
```

### Components
| Component | Details |
|-----------|---------|
| **Base Encoder** | `BAAI/bge-m3` (1024-dim), **frozen** — all params `requires_grad=False` |
| **Adapter (A_ψ)** | Residual bottleneck: `1024 → 512/128 → LayerNorm → GELU → Dropout → 1024 → + residual → L2Norm`. Zero-initialized up-projection so it starts as identity |
| **GRL** | Gradient Reversal Layer with schedule `α_p = 2/(1+exp(-η·p)) - 1` (η=10) |
| **Discriminator (D_φ)** | `1024 → 512 → ReLU → Dropout → 256 → ReLU → Dropout → 1` (logits). Sanitized version centers vectors to zero component mean before classifying |
| **Domain Labels** | `y=0`: informal queries; `y=1`: formal queries (+ optionally positive docs) |

### Loss Function
```
L_total = L_InfoNCE + γ · L_domain + μ · L_preserve
```

- **L_InfoNCE**: Contrastive alignment between adapted informal query and its positive doc (temp τ=0.05). In-batch + mined hard negatives. False positives masked.
- **L_domain**: Class-balanced BCE on domain logits (0 for adapted informal, 1 for formal targets)
- **L_preserve**: `mean(1 - cos(A(z_form), z_form))` — keeps formal queries unchanged by adapter (weight μ=0.5 default)

### Training Loop
1. **Data**: Cached frozen embeddings from `EmbeddingStore` (formal, informal, corpus, hard negatives)
2. **Optimizer**: AdamW (lr=1e-4, wd=0.01), cosine schedule with optional warmup
3. **Mixed Precision**: BF16 via `accelerate`
4. **Evaluation**: Every `--eval_every` steps (default 50) on dev split
5. **Checkpointing**: Save adapter + discriminator when dev nDCG@10 improves
6. **Logging**: TensorBoard + JSONL (`log.jsonl`) — loss components, α, disc accuracy, lr, train acc@1

### Key Hyperparameters (defaults)
```bash
--epochs 30 --batch_size 64 --lr 1e-4 --tau 0.05 --gamma 0.1 --mu 0.5 --eta 10.0
--bottleneck 128 --adapter zeroinit --discriminator sanitized
--eval_every 50 --mixed_precision bf16
```

---

## 2. Evaluation Pipeline (`eval.py`, `inference.py`)

### Metrics
- **nDCG@10**, **MRR@10**, **Recall@100** (binary relevance)
- Per-query metrics → mean ± std
- **Paired bootstrap** (1000 samples) for significance: mean diff, 95% CI, p-value

### Conditions Compared
| # | Condition | Description |
|---|-----------|-------------|
| 1 | Base BGE-M3 \| formal query | Upper bound (clean queries) |
| 2 | Base BGE-M3 \| informal query | Zero-shot baseline (no adaptation) |
| 3 | Query-DANN \| informal query | **Proposed method** |
| 4 | Query-DANN \| formal query | Robustness check (adapter shouldn't hurt clean queries) |
| + | Extra checkpoints | Ablations via `--extra name=path` |

### Evaluation Flow (`Evaluator` class)
1. Load `EmbeddingStore` with pre-computed frozen embeddings
2. Restrict qrels to docs present in corpus
3. `dense_search`: exact inner-product top-k (cosine since all L2-normalized)
4. Run three query embeddings through retriever:
   - `store.formal` (base formal)
   - `store.informal` (base informal)
   - `head.adapt(store.informal)` (adapted informal)
5. Compute metrics per condition, print table, run bootstrap comparisons

### Output
```
MIRACL-id test split: 1000 queries, 500,000 passages
condition                                            nDCG@10  MRR@10  R@100
1. Base BGE-M3 | formal query (upper bound)          0.7234   0.6891  0.9123
2. Base BGE-M3 | informal query (zero-shot)          0.5123   0.4567  0.7890
3. Query-DANN  | informal query (proposed)           0.6987   0.6543  0.8901
   Query-DANN  | formal query (robustness)           0.7198   0.6845  0.9087

Paired bootstrap (1000 samples, two-sided; * = p < 0.05)
comparison                               metric    diff         95% CI           p
DANN informal vs zero-shot informal      nDCG@10  +0.1864  [+0.1652, +0.2078]  0.001 *
DANN formal vs base formal               nDCG@10  -0.0036  [-0.0121, +0.0048]  0.412
```

---

## 3. Visualization & Inference Tools

### A. `inference.py` — Sampled Demo (Default)
```bash
python3 inference.py --ckpt outputs/run_name/best --n 5 --split test
```

**Per-query report:**
- Cosine similarities:
  - `cos(informal, formal query)` — base vs adapted
  - `cos(informal, best gold passage)` — base vs adapted
  - `cos(formal, best gold passage)` — reference
  - `cos(base informal, adapted)` — how far adapter moved it
- First relevant rank: base informal | adapted informal | base formal
- Retrieval metrics table (nDCG@10, MRR@10, R@100)
- Top-k passages for base vs adapted (relevant marked with ✓)

**Aggregate summary:**
- Mean cosines across samples
- Hit@k counts
- Mean metrics table

### B. `inference.py` — Free-Text Mode
```bash
python3 inference.py --ckpt outputs/run_name/best --query "presiden pertama indonesia siapa sih"
```

Shows base vs adapted top-k for arbitrary user queries.

### C. `search.py` — Interactive Search
```bash
python3 search.py
query> presiden pertama indonesia siapa sih
query> :k 10           # change top-k
query> :mode both      # adapted | base | both (shows base rank for adapted results)
query> :full           # toggle full passage text
query> :quit
```

**Features:**
- Loads model + corpus once (~40s), then instant queries
- Side-by-side comparison with base model ranks
- Shows passages dropped by adapter with their original base rank
- One-shot mode with `--query` and `--json` output

### D. TensorBoard Logs
```
outputs/run_name/tb/
```
Tracked scalars:
- `loss/total`, `loss/infonce`, `loss/domain`, `loss/preserve`
- `grl/alpha` (GRL schedule progress)
- `lr`, `disc/accuracy`, `train/acc@1`
- `dev/adapted_*`, `dev/adapted_formal_*`, `dev/disc_accuracy`

---

## 4. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING (once)                       │
├─────────────────────────────────────────────────────────────────┤
│  corpus.json, queries.json, qrels.json, train_pairs.csv        │
│       │                                                         │
│       ▼                                                         │
│  EmbeddingStore.cache()  ──►  methods/cache/                    │
│  - corpus.pt       (500k × 1024)  L2-normalized doc embeddings │
│  - formal.pt       (N × 1024)     formal query embeddings      │
│  - informal.pt     (N × 1024)     informal query embeddings    │
│  - splits.json     train/dev/test query_id splits              │
│  - hard_negs.json  mined hard negative ranks per query         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING                                │
├─────────────────────────────────────────────────────────────────┤
│  TrainQueryDataset ──► DataLoader (triples: inf, form, pos, hn)│
│       │                                                         │
│       ▼                                                         │
│  QueryDANN head (adapter + GRL + discriminator)                │
│       │                                                         │
│       ▼                                                         │
│  L_InfoNCE + γ·L_domain + μ·L_preserve                          │
│       │                                                         │
│       ▼                                                         │
│  outputs/run_name/                                              │
│  ├─ best/adapter.pt, discriminator.pt, config.json             │
│  ├─ tb/  (TensorBoard events)                                   │
│  ├─ log.jsonl  (per-step JSONL)                                 │
│  └─ args.json  (hyperparameters)                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EVALUATION / INFERENCE                     │
├─────────────────────────────────────────────────────────────────┤
│  EmbeddingStore (frozen) + QueryDANN head (loaded from ckpt)   │
│       │                                                         │
│       ├─► eval.py          ──► metrics table + bootstrap tests │
│       ├─► inference.py     ──► per-query analysis + summary    │
│       └─► search.py        ──► interactive / one-shot search   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Frozen encoder** | Prevents catastrophic forgetting; document embeddings never drift |
| **Zero-init adapter** | Starts as identity → stable early training, no sudden retrieval collapse |
| **Residual + L2Norm** | Output stays on unit hypersphere; cosine similarity works correctly |
| **Sanitized discriminator** | Removes first-order stats (mean, norm) that GRL can exploit trivially |
| **GRL schedule (α)** | Starts near 0, ramps to 1; empirically the discriminator is never fooled, so this is a weak regulariser rather than proven domain alignment (`ABLATION_GAMMA0.md`) |
| **Preservation loss (μ)** | Explicitly penalizes moving formal queries; maintains upper-bound performance |
| **Cached embeddings** | Training only updates adapter + discriminator (tiny, fast); no encoder forward pass |
| **Hard negative mining** | Uses base model rankings to find challenging negatives per query |

---

## 6. Running Commands Quick Reference

```bash
# Train
cd bge_dann/methods/query_dann
python3 train.py --run_name my_run --gamma 0.1 --mu 0.5 --epochs 30

# Evaluate checkpoint
python3 -m methods.eval --ckpt methods/query_dann/outputs/my_run/best --split test --bootstrap 1000

# Sampled qualitative analysis
python3 inference.py --ckpt outputs/my_run/best --n 10 --split test

# Free-text demo
python3 inference.py --ckpt outputs/my_run/best --query "gunung tertinggi di dunia"

# Interactive search
python3 search.py --ckpt outputs/my_run/best

# TensorBoard
tensorboard --logdir outputs/my_run/tb
```