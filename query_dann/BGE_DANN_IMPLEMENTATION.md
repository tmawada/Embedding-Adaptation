# Query-DANN Adaptation Architecture for BGE-M3

## Project Goal
Build and train a **Query-DANN (Domain-Adversarial Neural Network)** architecture on top of the **BGE-M3** dense embedding model for Indonesian Retrieval-Augmented Generation (RAG). 

The goal is to map **informal/noisy Indonesian queries** into the same dense vector space as **formal Wikipedia queries and documents** without mutating or distorting the underlying document embeddings, eliminating the need for an LLM rewrite step during inference.

---

## Dataset Layout

The dataset files are located in the project root:

```
├── corpus.json          # Document corpus: {doc_id: text, title: ...}
├── queries.json         # MIRACL formal queries: {query_id: text}
├── qrels.json           # Relevance judgments: {query_id: {doc_id: relevance_score}}
├── miracl_pairs.json    # Paired positive/negative formal samples
└── train_pairs.csv      # Generated query alignment dataset
                         # Schema: [query_id, formal_query, generated_informal_query]
```

---

## Technical Specifications & Architecture

```
[ Informal Query (q_inf) ]  ──► [ Trainable Adapter A_ψ ] ──► z_q ──┬──► [ InfoNCE Loss ] ◄── z_d ◄── [ Frozen BGE-M3 Doc Encoder ]
                                                                     │
[ Formal Text (q_form / d) ] ──► [ Frozen BGE-M3 Encoder ] ───────► z_f ──┤
                                                                     │
                                                                     └──► [ GRL (λ) ] ──► [ Domain Discriminator D_ϕ ] ──► [ BCE Loss ]
```

### 1. Base Encoder
* Model: `BAAI/bge-m3` (Dense representation output, vector dimension $d = 1024$).
* Configuration: **Document Encoder path MUST remain frozen** to prevent catastrophic forgetting of general document representations.

### 2. Query Bottleneck Adapter ($A_\psi$)
* Attachment: Applied on top of BGE-M3's dense `[CLS]` (or mean-pooled) output vector for query inputs.
* Architecture:
  * Linear Layer: $1024 \rightarrow 512$
  * LayerNorm + GELU Activation + Dropout ($p = 0.1$)
  * Residual Linear Projection: $512 \rightarrow 1024$
  * Skip Connection: $z_{\text{out}} = \text{LayerNorm}(z_{\text{in}} + \text{Projection}(z_{\text{bottleneck}}))$

### 3. Gradient Reversal Layer (GRL)
Implement a custom PyTorch `torch.autograd.Function`:

```python
import torch

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)
```

### 4. Domain Discriminator ($D_\phi$)
* Input: 1024-dimensional dense vectors passing through the GRL.
* Architecture:
  * Linear($1024 \rightarrow 512$) $\rightarrow$ ReLU $\rightarrow$ Dropout($0.2$)
  * Linear($512 \rightarrow 256$) $\rightarrow$ ReLU $\rightarrow$ Dropout($0.2$)
  * Linear($256 \rightarrow 1$) $\rightarrow$ Sigmoid
* Target Labels:
  * $y = 0$: Informal Queries (`generated_informal_query` from `train_pairs.csv`)
  * $y = 1$: Formal Queries / Target Documents (`formal_query` or MIRACL positive docs)

---

## Optimization Objectives & Loss Function

Total training loss per step:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{InfoNCE}} + \gamma \cdot \mathcal{L}_{\text{domain}}$$

### 1. Task Loss ($\mathcal{L}_{\text{InfoNCE}}$)
Aligns the adapted informal query embedding $z_{q,\text{inf}}$ with its corresponding positive document embedding $z_{d^+}$:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\cos(z_{q,\text{inf}}, z_{d^+}) / \tau)}{\exp(\cos(z_{q,\text{inf}}, z_{d^+}) / \tau) + \sum_{d^- \in \mathcal{N}} \exp(\cos(z_{q,\text{inf}}, z_{d^-}) / \tau)}$$

* Temperature hyperparameter: $\tau = 0.02$
* Include in-batch hard negative documents from `miracl_pairs.json`.

### 2. Domain Loss ($\mathcal{L}_{\text{domain}}$)
Calculates Binary Cross-Entropy on domain classification:

$$\mathcal{L}_{\text{domain}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log D_\phi(\text{GRL}(z_i)) + (1 - y_i) \log(1 - D_\phi(\text{GRL}(z_i))) \right]$$

### 3. Dynamic GRL Scheduling ($\alpha$)
Scale $\alpha$ dynamically over training progress $p \in [0, 1]$ to stabilize early convergence:

$$\alpha_p = \frac{2}{1 + \exp(-\eta \cdot p)} - 1 \quad (\text{with } \eta = 10)$$

---

## Step-by-Step Implementation Instructions

1. **Data Pipeline Setup (`dataset.py`)**:
   * Load `train_pairs.csv` and join with `miracl_pairs.json` via `query_id`.
   * Construct triples: `(informal_query, formal_query, positive_doc_text, negative_doc_text)`.
   * Implement a PyTorch `DataLoader` with dynamic batching.

2. **Model Definition (`model.py`)**:
   * Instantiate `BAAI/bge-m3` using `transformers` or `sentence-transformers`.
   * Freeze all base Transformer parameters: `for p in bge_model.parameters(): p.requires_grad = False`.
   * Implement `QueryAdapter`, `GradientReversalLayer`, and `DomainDiscriminator` modules.

3. **Training Loop (`train.py`)**:
   * Optimizer: AdamW on $A_\psi$ and $D_\phi$ parameters ($\text{lr} = 2\text{e-}4$, weight decay = $0.01$).
   * Calculate embeddings:
     * $z_{q,\text{inf}} = A_\psi(\text{Encoder}(q_{\text{informal}}))$
     * $z_{d^+} = \text{Encoder}(d^+)$ *(No adapter applied)*
     * $z_{q,\text{form}} = \text{Encoder}(q_{\text{formal}})$ *(No adapter applied)*
   * Pass $z_{q,\text{inf}}$ (labeled $0$) and $z_{q,\text{form}}$ / $z_{d^+}$ (labeled $1$) to the GRL + Discriminator block.
   * Compute $\mathcal{L}_{\text{total}}$ and perform backward pass.

4. **Evaluation Engine (`eval.py`)**:
   * Evaluate retrieval quality on MIRACL using `qrels.json` and `corpus.json`.
   * Metrics to calculate: **nDCG@10**, **MRR@10**, and **Recall@100**.
   * Compare three baseline conditions:
     1. Unadapted Base BGE-M3 on `formal_query` (Upper bound baseline).
     2. Unadapted Base BGE-M3 on `generated_informal_query` (Zero-shot baseline).
     3. **Query-DANN Adapted BGE-M3** on `generated_informal_query` (Proposed method).

---

## Code Quality Requirements

* **Frameworks**: PyTorch, Hugging Face `transformers`, `accelerate` (for mixed-precision FP16/BF16 training).
* **Logging**: Track loss components ($\mathcal{L}_{\text{InfoNCE}}$, $\mathcal{L}_{\text{domain}}$), $\alpha$ values, and evaluation metrics using TensorBoard or WandB.
* **Checkpointing**: Save adapter weights ($A_\psi$) and discriminator weights ($D_\phi$) whenever evaluation nDCG@10 reaches a new high.