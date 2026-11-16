# Model Pipeline Summarization — `bge_dann`

## Overview & Task Definition

This codebase implements and evaluates **Informal-to-Formal Query Adaptation for Dense Information Retrieval** on the **MIRACL-id** (Indonesian) benchmark. Colloquial or informal user queries (e.g., product search, everyday spoken Indonesian) degrade retrieval performance when directly matched against formal corpus passages using standard dense retrievers.

To ensure fair comparison, **all methods share a single frozen 500,000-passage index** and a frozen **BAAI/bge-m3** dense encoder. The central objective is adapting query representations in dense cosine space without re-indexing the corpus.

**Terminology:** Pipelines 1–2 are *paired query-style adaptation* — they train on informal↔formal query pairs together with labelled positives and mined hard negatives. "Unsupervised" / UDA refers only to the TSDAE full-encoder training (unlabelled text) and Pipeline 5 (`uda_query_dann/`, unpaired informal queries).

---

## 📁 Project Layout

```
bge_dann/
├── preprocessing/dataset.py          # data loading, EmbeddingStore, training triples
├── data/                             # raw data
├── methods/
│   ├── model.py eval.py encoders.py  #   shared library: encoders/adapters/discriminators, evaluator + metrics
│   ├── prepare.py                    #   builds cache/ (run once, before any method)
│   ├── cache/                        #   shared frozen BGE-M3 embedding cache, read by every method
│   ├── query_dann/                   #   Pipeline 1
│   ├── query-dann-v2/                #   Pipeline 1 variant: symmetric GRL (formal queries also adapted)
│   ├── query_distillation/           #   Pipeline 2
│   ├── tsdae_sapt/                   #   Pipeline 3 (reviews)
│   ├── tsdae_sapt_queries/           #   Pipeline 3 (task queries)
│   ├── stacked/                      #   Pipeline 4
│   ├── uda_query_dann/               #   Pipeline 5
│   ├── query_infonce/                #   InfoNCE + TSDAE encoder fine-tuning
│   ├── embed_transform/              #   closed-form / MLP informal->formal transforms
│   └── llm_rewrite/                  #   text-level LLM normalisation baseline (no embedding training)
├── multi_seed/                       # cross-method seed-variance experiment
└── evaluate_selected_models.py       # cross-model test-set table
```

**Training outputs stay inside the method directory.** Every run of `methods/<method>/train.py` writes
`methods/<method>/outputs/<run_name>/` with `args.json`, `log.jsonl`, TensorBoard `tb/` and the `best/` checkpoint.
The full-encoder methods save the encoder together with its `log.jsonl` to `methods/<method>/outputs/<name>/`
(`tsdae_sapt`, `query_infonce`); `embed_transform` writes `methods/embed_transform/outputs/transforms/`.
(`llm_rewrite` and `multi_seed` still write to `runs/`.)
Build the shared cache (`methods/cache/`) once with `python3 methods/prepare.py`, then e.g.
`python3 methods/query_dann/train.py --run_name my_run`.

---

## 🏗️ Architecture & Model Catalog

All query adapters operate on **1024-dimensional dense vectors** (`EMB_DIM = 1024`).

```
                    ┌───────────────────────────────────────────────┐
                    │               Base Architecture               │
                    └───────────────────────┬───────────────────────┘
                                            │
           ┌────────────────────────────────┴────────────────────────────────┐
           ▼                                                                 ▼
 ┌───────────────────┐                                             ┌───────────────────┐
 │ Base Encoder      │                                             │ Fine-Tuned Tower  │
 │ BAAI/bge-m3       │                                             │ TSDAE-SAPT        │
 │ (Frozen, [CLS])   │                                             │ (XLM-R / BGE-M3)  │
 └─────────┬─────────┘                                             └─────────┬─────────┘
           │                                                                 │
           ├──────────────────────────────┬──────────────────────────────────┤
           ▼                              ▼                                  ▼
 ┌──────────────────┐           ┌──────────────────┐               ┌──────────────────┐
 │ 1. Query-DANN    │           │ 2. Distillation  │               │ 3. Stacked       │
 │    Adapter       │           │    Adapter       │               │    (DANN + KD)   │
 └──────────────────┘           └──────────────────┘               └──────────────────┘
```

### 1. Base Encoder (`BGEM3Encoder` / `GenericEncoder`)
* **Location**: `methods/model.py`, `methods/encoders.py`
* **Model ID**: `BAAI/bge-m3` (1024-dim, XLM-RoBERTa architecture)
* **Pooling**: `[CLS]` token pooling followed by float32 L2-normalization, cast to `float16`.
* **Status**: Kept **frozen** (`requires_grad = False`, `.eval()`) during adapter training. Encodes all texts once via length-sorted dynamic batching (`methods/prepare.py`).

### 2. Query Adapter Modules (`ADAPTERS` in `methods/model.py`)
Lightweight bottleneck networks $A_\psi(\cdot)$ mapping input query representations $\mathbf{z}_{\text{in}} \in \mathbb{R}^{1024} \to \mathbf{z}_{\text{out}} \in \mathbb{R}^{1024}$.

* **`ZeroInitQueryAdapter` (`zeroinit`, Default)**:
  $$\mathbf{z}_{\text{out}} = \text{L2Norm}\left(\mathbf{z}_{\text{in}} + W_{\text{up}} \cdot \text{Dropout}(\text{GELU}(\text{LayerNorm}(W_{\text{down}} \cdot \mathbf{z}_{\text{in}})))\right)$$
  * $W_{\text{down}} \in \mathbb{R}^{128 \times 1024}$, $W_{\text{up}} \in \mathbb{R}^{1024 \times 128}$
  * $W_{\text{up}}$ and bias are **zero-initialized**, ensuring $A_\psi(\mathbf{z}) = \mathbf{z}$ at step 0 (exact identity mapping).
* **`QueryAdapter` (`layernorm`)**: Bottleneck dim 512 with **L2Norm** output (changed from LayerNorm for fair comparison).
* **`LinearQueryAdapter` (`linear`)**: Parameterized $1024 \times 1024$ linear transformation initialized to the identity matrix.
* **`IdentityQueryAdapter` (`identity`)**: Parameterless identity mapping used for zero-shot base model baselines.

### 3. Domain Discriminators (`DISCRIMINATORS` in `methods/model.py`)
Classifiers $D_\phi(\cdot)$ predicting domain logits from query representations.

* **`SanitizedDomainDiscriminator` (`sanitized`, Default)**:
  * Centers component mean to 0 and re-normalizes vectors before classification: $\mathbf{z} \leftarrow \text{L2Norm}(\mathbf{z} - \bar{\mathbf{z}})$. Prevents the discriminator from exploiting trivial mean norm shifts.
  * MLP: $\text{Linear}(1024 \to 512) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(512 \to 256) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(256 \to 1)$
* **`DomainDiscriminator` (`plain`)**: Standard MLP without component mean centering.

### 4. Gradient Reversal Layer (GRL)
* **Implementation**: `GradientReversalLayer` / `grad_reverse`
* **Behavior**: Identity during forward pass; negates gradients and scales by $\alpha_p$ during backward pass:
  $$\alpha_p = \frac{2}{1 + \exp(-\eta \cdot p)} - 1, \quad p = \frac{\text{step}}{\text{total\_steps}}, \quad \eta = 10.0$$

---

## 🔄 Training Pipelines & Objectives

### Pipeline 1: Query-DANN v2 (`methods/query_dann/train.py`)
*Main proposed method using paired informal/formal queries, contrastive retrieval and an adversarial domain term.*

The "bottleneck vs. no bottleneck" comparison is controlled by `--adapter zeroinit` (residual 128-dim bottleneck) versus `--adapter linear` (plain 1024×1024 map). `--adapter identity` is the no-adapter lower bound.

> **Adversarial-term finding:** at the default `gamma=0.1`, the discriminator's dev accuracy rises from 0.5 to ~0.95 and the domain loss falls from ~0.69 to ~0.15 — it is *not* fooled, so the representations are not domain-invariant. A `--gamma 0` control (seeds 42/84/123/256/512) scores **+0.0047 dev** and **+0.0005 test** nDCG@10 *better* than `gamma=0.1` on average, i.e. the GRL term gives no measurable benefit. The retrieval gain comes from InfoNCE + formal preservation; the GRL term is reported as a weak regulariser. Full report: `methods/query_dann/ABLATION_GAMMA0.md`, §Method 1 of `informal-query-embedding-adaptation-spec.md`.

**Symmetric-GRL variant (`methods/query-dann-v2/`):** formal queries also pass through the adapter before the discriminator, so GRL reaches the adapter from both domains. This fixes the asymmetry: `gamma=0.1` now beats `gamma=0` by +0.0051 dev / +0.0067 test nDCG@10 (5 seeds, 5/5 positive) and beats the original variant by +0.0072 test. D accuracy ends at ~0.82–0.85, so this is balanced adversarial regularisation, still not domain invariance. Same head architecture and checkpoint format; see `methods/query-dann-v2/ABLATION_SYMMETRIC_GRL.md`.

```
informal query ──> [BGE-M3] ──> z_inf ──> Adapter A_ψ ──> z_adapted ───┬──> Dense Search
                                                                       │
formal query   ──> [BGE-M3] ──> z_form ───> Discriminator D_φ <── GRL <┤
```

* **Loss Functions**:
  1. **InfoNCE Contrastive Loss** ($\tau=0.05$):
     $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\mathbf{z}_q \cdot \mathbf{z}_{\text{pos}} / \tau)}{\sum_{j} \exp(\mathbf{z}_q \cdot \mathbf{z}_{\text{cand}, j} / \tau)}$$
     *(Candidates include in-batch positives + mined hard negatives from ranks 10–100 with false-negative masking)*
  2. **Adversarial Domain Loss** ($\gamma=0.1$):
     $$\mathcal{L}_{\text{domain}} = \frac{1}{2} \text{BCEWithLogits}(D_\phi(\text{GRL}(\mathbf{z}_q, \alpha)), 0) + \frac{1}{2} \text{BCEWithLogits}(D_\phi(\text{GRL}(\mathbf{z}_{\text{target}}, \alpha)), 1)$$
  3. **Formal Preservation Loss** ($\mu=0.5$):
     $$\mathcal{L}_{\text{preserve}} = \frac{1}{B} \sum_{i=1}^B (1 - A_\psi(\mathbf{z}_{\text{form}, i}) \cdot \mathbf{z}_{\text{form}, i})$$
  4. **Total Loss**:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{InfoNCE}} + \gamma \mathcal{L}_{\text{domain}} + \mu \mathcal{L}_{\text{preserve}}$$

---

### Pipeline 2: Query Distillation (`models/query_distillation/train.py`)
*Supervised paired query adaptation without adversarial domain training.*

* **Teacher**: Base BGE-M3 on formal queries ($\mathbf{z}_{\text{form}}$)
* **Student**: Adapter $A_\psi$ on informal queries ($\mathbf{z}_q = A_\psi(\mathbf{z}_{\text{inf}})$)
* **Distillation Loss Modes**:
  * **Embedding Distillation** (`--distill embed`): $\mathcal{L}_{\text{KD}} = 1 - \mathbf{z}_q \cdot \mathbf{z}_{\text{form}}$
  * **Score Distillation** (`--distill score`):
    $$\mathcal{L}_{\text{KD}} = \text{KL}\left(\text{softmax}\left(\frac{\mathbf{z}_{\text{form}} \cdot C_k}{\tau_{\text{kd}}}\right) \;\middle\|\; \text{softmax}\left(\frac{\mathbf{z}_q \cdot C_k}{\tau_{\text{kd}}}\right)\right)$$
    *(over top $k=100$ corpus candidates $C_k$ retrieved by the teacher)*
* **Total Loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{KD}} + w_{\text{infonce}} \mathcal{L}_{\text{InfoNCE}} + \mu \mathcal{L}_{\text{preserve}}$

---

### Pipeline 3: TSDAE-SAPT (Sentence-Adaptive Pre-Training) (`models/tsdae_sapt/train_sapt.py`)
*Stage 1 encoder fine-tuning on informal domain text.*

```
noisy review (60% word deletion) ──> BGE-M3 Encoder ──> [CLS] bottleneck
                                                              │
                                                        Cross-Attention
                                                              ▼
original review ◄───────────────────────────────── Transformer Decoder
```

* **Objective**: Reconstruct uncorrupted original text from 60% deleted word noise using the `[CLS]` bottleneck vector.
* **Encoder status**: **Fully fine-tuned** (unlike the frozen adapter setups). Decoder parameters are discarded post-training.
* **Data**: 4,911 unique informal Indonesian product reviews from Tokopedia (`prdect-id.csv`) or 2,891 task informal queries (`models/tsdae_sapt_queries/`).
* **Loss**: Token cross-entropy reconstruction loss.

---

### Pipeline 4: Stacked DANN + Distillation (`models/stacked/train.py`)
*Combines Stage 1 (TSDAE encoder tuning) and Stage 2 (Query-DANN + Score Distillation) into a unified end-to-end framework.*

* **Total Loss**:
  $$\mathcal{L}_{\text{stacked}} = \mathcal{L}_{\text{InfoNCE}} + \gamma \mathcal{L}_{\text{domain}} + \mu \mathcal{L}_{\text{preserve}} + \lambda_{\text{kd}} \mathcal{L}_{\text{KD\_score}}$$

---

### Pipeline 5: UDA Query-DANN (`models/uda_query_dann/train.py`)
*Unsupervised domain adaptation where informal queries have **no labels** and are **unpaired** with formal queries.*

* **Source Domain**: Formal queries with relevance labels $\to$ InfoNCE task loss
* **Target Domain**: Informal queries without labels or pairs $\to$ Discriminator target
* **Splits**: `disjoint` (training query set split 50/50 across domains) or `overlap` (unpaired sampling across full set).

---

### Pipeline 6: LLM Query Normalisation Baseline (`llm_rewrite/`)
*Text-level normalization baseline.*

* Informal queries are rewritten into formal Indonesian text via an LLM (`gemini-3.5-flash-lite`), then encoded using the unchanged base BGE-M3 model.
* Serves as a baseline to benchmark latency vs. adapter inference (~1000× faster for adapters).

---

## 📊 Preprocessing, Data & Evaluation Protocol

### 1. Data Caching Pipeline (`methods/prepare.py`)
* `formal_emb.npy`: $(2891, 1024)$ float16 embeddings of formal queries.
* `informal_emb.npy`: $(2891, 1024)$ float16 embeddings of informal queries.
* `corpus_emb.npy`: $(500000, 1024)$ memory-mapped float16 corpus embeddings.
* `hard_negs.json`: Top-200 base-model hard negative passage IDs mined per query.
* `splits.json`: Disjoint split (75% Train, 10% Dev, 15% Test) at the query level to prevent leakage.

### 2. Evaluation Framework (`methods/eval.py` & `comparison/compare_all.py`)
* **Metrics**: `nDCG@10` (primary), `MRR@10`, `Recall@100`
* **Search Engine**: Exact inner-product matrix multiplication over all 500k passage embeddings.
* **Significance Testing**: Paired bootstrap resampling ($N = 1000$ iterations, two-sided $p$-value with $95\%$ confidence intervals).

---

## ⚙️ Hyperparameter Summary Table

| Parameter | Value | Scope | Description |
|---|---|---|---|
| `EMB_DIM` | 1024 | Global | Hidden dimension of BGE-M3 embeddings |
| `bottleneck` | 128 | Adapter | Bottleneck dimension ($1024 \to 128 \to 1024$) |
| `adapter_dropout` | 0.1 | Adapter | Dropout rate inside adapter |
| `disc_dropout` | 0.2 | Discriminator | Dropout rate inside discriminator |
| `gamma` ($\gamma$) | 0.1 / 1.0 | Query-DANN | Weight of domain adversarial loss |
| `mu` ($\mu$) | 0.5 | Query-DANN / KD | Weight of formal preservation loss |
| `tau` ($\tau$) | 0.05 | InfoNCE | Contrastive loss temperature |
| `kd_tau` | 0.05 | Distillation | Softmax temperature for teacher/student scores |
| `eta` ($\eta$) | 10.0 | GRL | Gradient Reversal Layer schedule multiplier |
| `batch_size` | 64 | Training | Training batch size for adapters |
| `lr` | $1 \times 10^{-4}$ | Training | AdamW initial learning rate (cosine schedule to $1 \times 10^{-6}$) |
| `epochs` | 30 (~1020 steps) | Training | Total adapter training epochs |
| `del_ratio` | 0.6 (60%) | TSDAE-SAPT | Word deletion probability for noise generation |
| `sapt_lr` | $3 \times 10^{-5}$ | TSDAE-SAPT | Constant learning rate for encoder fine-tuning |
