# Model Pipeline Summarization — `bge_dann`

## Overview & Task Definition

This codebase implements and evaluates **Informal-to-Formal Query Adaptation for Dense Information Retrieval** on the **MIRACL-id** (Indonesian) benchmark. Colloquial or informal user queries (e.g., product search, everyday spoken Indonesian) degrade retrieval performance when directly matched against formal corpus passages using standard dense retrievers.

To ensure fair comparison, **all methods share a single frozen 500,000-passage index** and a frozen **BAAI/bge-m3** dense encoder. The central objective is adapting query representations in dense cosine space without re-indexing the corpus.

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
* **Location**: `model.py`, `encoder_comparison/encoders.py`
* **Model ID**: `BAAI/bge-m3` (1024-dim, XLM-RoBERTa architecture)
* **Pooling**: `[CLS]` token pooling followed by float32 L2-normalization, cast to `float16`.
* **Status**: Kept **frozen** (`requires_grad = False`, `.eval()`) during adapter training. Encodes all texts once via length-sorted dynamic batching (`prepare.py`).

### 2. Query Adapter Modules (`ADAPTERS` in `model.py`)
Lightweight bottleneck networks $A_\psi(\cdot)$ mapping input query representations $\mathbf{z}_{\text{in}} \in \mathbb{R}^{1024} \to \mathbf{z}_{\text{out}} \in \mathbb{R}^{1024}$.

* **`ZeroInitQueryAdapter` (`zeroinit`, Default)**:
  $$\mathbf{z}_{\text{out}} = \text{L2Norm}\left(\mathbf{z}_{\text{in}} + W_{\text{up}} \cdot \text{Dropout}(\text{GELU}(\text{LayerNorm}(W_{\text{down}} \cdot \mathbf{z}_{\text{in}})))\right)$$
  * $W_{\text{down}} \in \mathbb{R}^{128 \times 1024}$, $W_{\text{up}} \in \mathbb{R}^{1024 \times 128}$
  * $W_{\text{up}}$ and bias are **zero-initialized**, ensuring $A_\psi(\mathbf{z}) = \mathbf{z}$ at step 0 (exact identity mapping).
* **`QueryAdapter` (`layernorm`)**: Bottleneck dim 512 with **L2Norm** output (changed from LayerNorm for fair comparison).
* **`LinearQueryAdapter` (`linear`)**: Parameterized $1024 \times 1024$ linear transformation initialized to the identity matrix.
* **`IdentityQueryAdapter` (`identity`)**: Parameterless identity mapping used for zero-shot base model baselines.

### 3. Domain Discriminators (`DISCRIMINATORS` in `model.py`)
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

### Pipeline 1: Query-DANN v2 (`query_dann/train.py`)
*Main proposed method using paired informal/formal queries and adversarial domain adaptation.*

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

### Pipeline 2: Query Distillation (`query_distillation/train.py`)
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

### Pipeline 3: TSDAE-SAPT (Sentence-Adaptive Pre-Training) (`tsdae_sapt/train_sapt.py`)
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
* **Data**: 4,911 unique informal Indonesian product reviews from Tokopedia (`prdect-id.csv`) or 2,891 task informal queries (`tsdae_sapt_queries/`).
* **Loss**: Token cross-entropy reconstruction loss.

---

### Pipeline 4: Stacked DANN + Distillation (`stacked/train.py`)
*Combines Stage 1 (TSDAE encoder tuning) and Stage 2 (Query-DANN + Score Distillation) into a unified end-to-end framework.*

* **Total Loss**:
  $$\mathcal{L}_{\text{stacked}} = \mathcal{L}_{\text{InfoNCE}} + \gamma \mathcal{L}_{\text{domain}} + \mu \mathcal{L}_{\text{preserve}} + \lambda_{\text{kd}} \mathcal{L}_{\text{KD\_score}}$$

---

### Pipeline 5: UDA Query-DANN (`uda_query_dann/train.py`)
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

### 1. Data Caching Pipeline (`prepare.py`)
* `formal_emb.npy`: $(2891, 1024)$ float16 embeddings of formal queries.
* `informal_emb.npy`: $(2891, 1024)$ float16 embeddings of informal queries.
* `corpus_emb.npy`: $(500000, 1024)$ memory-mapped float16 corpus embeddings.
* `hard_negs.json`: Top-200 base-model hard negative passage IDs mined per query.
* `splits.json`: Disjoint split (75% Train, 10% Dev, 15% Test) at the query level to prevent leakage.

### 2. Evaluation Framework (`eval.py` & `comparison/compare_all.py`)
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
