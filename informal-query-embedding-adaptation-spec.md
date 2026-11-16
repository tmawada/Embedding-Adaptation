# Technical Specification & Implementation Reference: Informal-to-Formal Query Embedding Adaptation for Indonesian Dense Retrieval

## 1. Executive Summary & Problem Formulation

### 1.1 Context & Motivation
In Indonesian Information Retrieval (IR) and E-commerce search, user queries are frequently written in colloquial, informal, or abbreviated text (e.g., *"yg"*, *"gmn"*, *"dgn"*, *"klo"*, *"cr"*, *"dftr"*). Matching these informal queries directly against formal document indices (e.g., news, encyclopedias, product catalogs) using standard dense retrievers leads to significant performance degradation due to query-document vocabulary and feature distribution shifts.

Rather than relying on text-level LLM rewriting (an extra API call per query) or static dictionary normalization, this research framework adapts query representations **directly inside the dense vector space** ($\mathbb{R}^{1024}$). By appending lightweight adapters to a frozen base dense encoder (**BAAI/bge-m3** or **intfloat/multilingual-e5-large-instruct**), informal query representations are mapped into the formal document embedding space in sub-millisecond time (measured 0.14 ms per query for the bottleneck adapter, on top of the ~7 ms query encoding that every pipeline pays; see Method 5) without modifying or re-indexing the underlying 500,000-passage corpus index.

### 1.2 Conceptual Foundations in Prior Research (ADAPT-DANN)
This framework builds on the parameter-efficient adaptation principles introduced in **ADAPT-DANN** (*Sandriago et al., ICDEES 2026*), applied here to **paired informal-to-formal query-style adaptation** rather than to unpaired UDA:
1. **Targeting Compounding Failure Modes**: ADAPT-DANN established that vocabulary gaps, feature distribution shifts, and catastrophic forgetting must be handled explicitly through dedicated mechanisms.
2. **Frozen Encoder Backbone & Lightweight Adapters**: Freezing the primary encoder and updating <0.30% of total parameters in adapter modules prevents catastrophic forgetting of pre-trained semantic representations. Here the bottleneck adapter has 263,552 parameters (~0.05% of BGE-M3's ~568M) and the linear adapter 1,049,600 (~0.18%).
3. **Adversarial & Alignment Constraints**: A Gradient Reversal Layer (GRL) and a formal-vector preservation regulariser are included as constraints. The adversarial term is a *weak regulariser* at the default $\gamma=0.1$ (see the ablation note in Method 1); it is **not** evidence of domain-invariant representations. The genuinely unpaired, label-free UDA setting is isolated in Pipeline 5 (`methods/uda_query_dann/`).

**Terminology.** Pipelines 1–2 use informal↔formal query pairs, labelled positive passages and mined hard negatives; they are therefore described as *paired query adaptation*, not unsupervised domain adaptation. "Unsupervised" applies to the TSDAE full-encoder training (no relevance labels, unlabelled text only) and to Pipeline 5.

---

## 2. System Architecture & Base Encoders

```
                                  ┌───────────────────────────────────────────────┐
                                  │             Base Retriever Towers             │
                                  │   BAAI/bge-m3  OR  mE5-large-instruct         │
                                  │             (1024-dim, Frozen)                │
                                  └───────────────────────┬───────────────────────┘
                                                          │
                       ┌──────────────────────────────────┴──────────────────────────────────┐
                       ▼                                                                     ▼
             ┌───────────────────┐                                                 ┌───────────────────┐
             │ Informal Query    │                                                 │ Formal Query      │
             │ Vector z_inf      │                                                 │ Vector z_form     │
             └─────────┬─────────┘                                                 └─────────┬─────────┘
                       │                                                                     │
                       ▼                                                                     │
            ┌─────────────────────┐                                                          │
            │ Query Adapter A_ψ   │                                                          │
            │ (ZeroInit Bottleneck│                                                          │
            └──────────┬──────────┘                                                          │
                       │                                                                     │
            z_adapted  ▼                                                                     ▼
            ┌────────────────────────────────────────────────────────────────────────────────┐
            │                     Loss Objectives & Alignment Framework                      │
            │  • InfoNCE Task Loss (Karpukhin et al., 2020)                                  │
            │  • Score Distillation KL Loss (Ren et al., 2021; Zhuang & Zuccon, 2022)        │
            │  • Adversarial Domain BCE + GRL (Ganin et al., 2016; Sandriago et al., 2026)   │
            │  • Formal Vector Preservation Regularizer (Taigman et al., 2017)               │
            └────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Base Encoders

1. **BAAI/bge-m3 (Main)**
   * **Architecture**: XLM-RoBERTa-based backbone, output dimension $d = 1024$.
   * **Pooling Strategy**: `[CLS]` token representation followed by Float32 L2-normalization.
   * **Input Lengths**: queries truncated at 64 tokens, passages at 512; no prefixes.
2. **intfloat/multilingual-e5-large-instruct (mE5 Alternative, `--encoder me5`)**
   * **Architecture**: XLM-RoBERTa-large backbone, output dimension $d = 1024$.
   * **Query Formatting**: Every query (formal and informal) is prefixed with the model-card instruction `"Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery: "`. Passages are encoded without a prefix.
   * **Pooling Strategy**: Mean pooling over non-padding tokens followed by Float32 L2-normalization.
   * **Input Lengths**: queries truncated at 128 tokens (the instruction costs ~15), passages at 512.
3. **sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (third encoder, `--encoder mpnet`)**
   * **Architecture**: XLM-RoBERTa-base student (12 layers, ~278M parameters), output dimension $d = 768$.
   * **Training**: multilingual knowledge distillation of the English paraphrase-mpnet-base-v2 teacher onto parallel data for 50+ languages, Indonesian included (Reimers & Gurevych, EMNLP 2020). Released before MIRACL existed, so — unlike BGE-M3 and mE5 — it was **never trained on MIRACL or Mr. TyDi**: the control for the contamination caveat in §2.2.
   * **Pooling Strategy**: Mean pooling over non-padding tokens followed by Float32 L2-normalization; no prefixes.
   * **Input Lengths**: queries and passages truncated at 128 tokens, the model's released limit.
   * **Adapters**: the same architectures at its width ($768 \to 128 \to 768$ bottleneck; $768 \times 768$ linear). Trainers infer $d$ from the cache and store it in each checkpoint's `config.json`.
4. **Frozen vs. fine-tuned**: In the adapter methods (Methods 1–2) the encoder is kept **strictly frozen** (`requires_grad = False`, `.eval()`). The full-encoder methods (Methods 3–4) fine-tune a copy of the encoder that is used for **queries only**; passages always come from the frozen base encoder, so the index is never re-encoded.

Each encoder has its own frozen index and embedding cache with the same layout (`methods/cache` for BGE-M3, `methods/cache/me5_large_instruct` for mE5, `methods/cache/multilingual_mpnet` for multilingual-mpnet); every method trains and is scored on its encoder's cache.

### 2.2 Shared Benchmark Dataset & Index
* **Corpus**: **MIRACL-id** (Indonesian subset of the Multilingual Information Retrieval Across Countries and Languages benchmark).
* **Index**: A 500,000-passage subsample of the MIRACL-id corpus, encoded once per encoder into a memory-mapped Float16 dense index ($500{,}000 \times 1024$).
* **Queries**: The 4,071 queries of the official MIRACL-id **train** split, each paired with a generated informal version (`data/train_pairs.csv`). 216 queries with no relevant passage in the 500k index are excluded, leaving **3,855 query pairs**.
* **Query Splits**: The 3,855 pairs are split disjointly at query level (75% Train, 10% Dev, 15% Test = 2,891 / 386 / 578, seed 42). Dev and test queries that repeat an earlier question are then dropped (see *Duplicate Questions*), giving **2,891 / 372 / 553**. No dev or test question, formal or informal, occurs in training.
* **Hard Negatives**: Mined per encoder from that encoder's own ranking of the formal query (depth 200, labelled positives removed).
* **Contamination Caveat**: Both encoders list MIRACL and Mr. TyDi training data in their supervised fine-tuning. The formal test queries were therefore probably seen by the encoders, which inflates formal scores and may overstate the formal-to-informal drop. Comparisons between methods remain fair (all share the same exposure), but absolute numbers are not comparable to published MIRACL results.
* **Duplicate Questions**: The MIRACL-id train split repeats some questions under different ids. Before deduplication, 24 test and 12 dev queries had a train "twin" with the same formal or informal text (almost always with the same relevant passage), and one question repeated inside dev and one inside test. `preprocessing.dataset.drop_twins` removes these 39 dev / test queries (14 dev, 25 test) and keeps train unchanged; their ids are listed in `splits.json` under `twins_excluded` (`methods/prepare.py --drop_twins true`, the default; `methods/drop_twins.py` applied it to the existing caches). Measured before the removal, the twins mainly inflated Query-DANN (bottleneck) on BGE-M3: after retraining on the deduplicated splits its informal gain is +1.26 pts ($p = 0.060$, no longer significant; +1.76 before), while the distillation gains barely moved (+3.79 → +3.64, linear).
* **References**: MIRACL — Zhang et al. (TACL 2023); BGE-M3 — Chen et al. (Findings of ACL 2024); mE5 — Wang et al. (2024).

---

## 3. Detailed Method Specifications & Reference Implementations

### Method 1: Query-DANN v2 (Adversarial Domain Adaptation)

#### Overview
Combines **paired** informal/formal contrastive retrieval with an adversarial domain term implemented via a Gradient Reversal Layer (GRL). Training uses the informal↔formal query pairs, labelled positive passages and mined hard negatives; the downstream TSDAE variant and Pipeline 5 are the label-free / unpaired settings. The adapter $A_\psi$ is optimised for dense retrieval performance while the GRL term applies a weak pressure to confuse the domain discriminator $D_\phi$ (see the ablation note below).

```
informal query ──> [Base Encoder] ──> z_inf ──> Adapter A_ψ ──> z_adapted ───┬──> Dense Search Index
                                                                              │
formal query   ──> [Base Encoder] ──> z_form ───> Discriminator D_φ <── GRL <─┤
```

#### Mathematical Formulation

1. **Zero-Initialized Bottleneck Query Adapter ($A_\psi$)**:
   $$\mathbf{z}_{\text{out}} = \text{L2Norm}\left(\mathbf{z}_{\text{in}} + W_{\text{up}} \cdot \text{Dropout}(\text{GELU}(\text{LayerNorm}(W_{\text{down}} \cdot \mathbf{z}_{\text{in}})))\right)$$
   Where $W_{\text{down}} \in \mathbb{R}^{128 \times 1024}$, $W_{\text{up}} \in \mathbb{R}^{1024 \times 128}$. Setting $W_{\text{up}} = 0$ and $b_{\text{up}} = 0$ at initialization guarantees $A_\psi(\mathbf{z}) = \mathbf{z}$ at step 0 (identity mapping).

   **Adapter variants** (shared by Methods 1 and 2):
   * **Bottleneck** (default, `--adapter zeroinit`): the zero-initialized residual adapter above, 263,552 parameters.
   * **Linear** (`--adapter linear`): a single map $\mathbf{z}_{\text{out}} = \text{L2Norm}(W\mathbf{z}_{\text{in}} + b)$ with $W \in \mathbb{R}^{1024 \times 1024}$ initialized to the identity and $b = 0$ (1,049,600 parameters). No bottleneck, residual branch, LayerNorm, GELU or dropout; it isolates what the adapter architecture contributes.

2. **Sanitized Domain Discriminator ($D_\phi$)**:
   To prevent $D_\phi$ from exploiting trivial first-order statistics (component mean, norm), each vector is centered on its own component mean $\bar{z} = \frac{1}{1024}\sum_{c} z_c$ and re-normalized prior to classification:
   $$\mathbf{z} \leftarrow \text{L2Norm}(\mathbf{z} - \bar{z}\,\mathbf{1})$$
   $$D_\phi(\mathbf{z}) = \text{Linear}_{256 \to 1}\left(\text{Dropout}_{0.2}\left(\text{ReLU}\left(\text{Linear}_{512 \to 256}\left(\text{Dropout}_{0.2}\left(\text{ReLU}\left(\text{Linear}_{1024 \to 512}(\mathbf{z})\right)\right)\right)\right)\right)\right)$$
   $D_\phi$ outputs a logit; the sigmoid is folded into `BCEWithLogits`.

3. **Gradient Reversal Layer (GRL) & Schedule**:
   GRL acts as identity during forward pass and negates/scales gradients during backpropagation by $\alpha_p$:
   $$\alpha_p = \frac{2}{1 + \exp(-\eta \cdot p)} - 1, \quad p = \frac{\text{current\_step}}{\text{total\_steps}}, \quad \eta = 10.0$$

4. **Multi-Objective Loss Function**:
   $$\mathcal{L}_{\text{Query-DANN}} = \mathcal{L}_{\text{InfoNCE}} + \gamma \mathcal{L}_{\text{domain}} + \mu \mathcal{L}_{\text{preserve}}$$

   * **Task InfoNCE Loss ($\tau = 0.05$)**:
     $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\mathbf{z}_q \cdot \mathbf{z}_{\text{pos}} / \tau)}{\sum_{j} \exp(\mathbf{z}_q \cdot \mathbf{z}_{\text{cand}, j} / \tau)}$$
     *Candidates*: one pool of $2B$ passages shared by the whole batch, namely the $B$ positives (one labelled positive per query) and $B$ hard negatives (one per query, re-sampled every epoch from ranks 10–100 of the base model's ranking of the formal query). A candidate that is a labelled positive of query $i$ but not its own target is masked out of query $i$'s denominator (false-negative masking).

   * **Adversarial Domain Loss ($\gamma = 0.1$)**:
     $$\mathcal{L}_{\text{domain}} = \frac{1}{2} \text{BCEWithLogits}(D_\phi(\text{GRL}(\mathbf{z}_q, \alpha_p)), 0) + \frac{1}{2} \text{BCEWithLogits}(D_\phi(\text{GRL}(\mathbf{z}_{\text{form}}, \alpha_p)), 1)$$

   * **Formal Preservation Regularizer ($\mu = 0.5$)**:
     $$\mathcal{L}_{\text{preserve}} = \frac{1}{B} \sum_{i=1}^B \left(1 - A_\psi(\mathbf{z}_{\text{form}, i}) \cdot \mathbf{z}_{\text{form}, i}\right)$$

#### Ablation note: what the adversarial term actually does
At the harmonised default ($\gamma = 0.1$, $\mu = 0.5$, 1,350 steps, `--adapter zeroinit`), the training logs show the domain loss falling from $\ln 2 \approx 0.693$ to $\approx 0.15$ while dev discriminator accuracy rises from 0.5 to $\approx 0.95$. The discriminator therefore **wins**: the adapted representations remain distinguishable from formal queries, and domain invariance is not achieved. Two structural reasons: (i) $\gamma=0.1$ gives the adversarial branch only a few percent of the total gradient, and (ii) the target branch uses the cached $\mathbf{z}_{\text{form}}$ directly, so GRL has no gradient path to the adapter there — the discriminator is trained on both domains while the adapter is pushed only on the informal side. A control with `--gamma 0` (seeds 42/84/123/256/512) scores on average **+0.0047 better dev** and **+0.0005 better test** informal nDCG@10 than `γ = 0.1` (test 95% CI of the difference `[−0.0044, +0.0040]`), so the adversarial term provides no measurable benefit. The reported gain is attributed to InfoNCE + preservation, and the GRL term is described as a weak adversarial regulariser rather than the mechanism of adaptation. See `methods/query_dann/ABLATION_GAMMA0.md`.

**Follow-up (symmetric GRL).** Passing formal queries through the adapter before the discriminator equalises the gradient flow and flips this result: in `methods/query-dann-v2/`, `γ = 0.1` gives **+0.0051 dev / +0.0067 test** over `γ = 0` (5 seeds, 5/5 positive). D accuracy ends at ~0.82–0.85 rather than ~0.95, so this supports the asymmetry diagnosis but is still balanced adversarial regularisation, not domain invariance. See `methods/query-dann-v2/ABLATION_SYMMETRIC_GRL.md`.

#### Provenance
Each component checked against the cited paper (2026-10-19); the code is verified numerically by `audit/crosscheck.py` (report: `audit/outputs/crosscheck_report.md`). *Faithful* = the published method and setting; *adapted* = from the paper with the stated change; *own design* = no published source, a choice of this work.

| Component | Source (section) | Status |
| :--- | :--- | :--- |
| Gradient reversal: identity forward, gradient × $-\alpha$ backward | Ganin & Lempitsky 2015 (§3); Ganin et al. 2016 (§4.2) | Faithful |
| Schedule $\alpha_p = 2/(1+e^{-10p}) - 1$ | Ganin et al. 2016 (§5.2.2, $\gamma = 10$) | Faithful |
| Domain loss: balanced binary cross-entropy, weight $\gamma = 0.1$ | Ganin et al. 2016 (§4.1) | Adapted: adversarial weight on the adapter is $\gamma\alpha_p \le 0.1$ instead of $\alpha_p$ |
| Domain-adversarial training for dense retrieval | Xin et al. 2022, MoDIR | Adapted: domains are informal vs. formal queries, and only a query adapter on a frozen encoder is trained |
| Sanitized (centred, re-normalised) discriminator | — | Own design: removes a component-mean shortcut found in this project |
| InfoNCE = negative log-likelihood of the positive | van den Oord et al. 2018 (§2.3); Karpukhin et al. 2020 (§3.2) | Faithful |
| In-batch positives + one hard negative per query, shared by the batch | Karpukhin et al. 2020 (§3.2, §5.2); Xiong et al. 2021, ANCE | Adapted: negatives mined with the dense base model instead of BM25 |
| $\tau = 0.05$ on cosine similarity | Gao et al. 2021, SimCSE (Appendix D) | Faithful |
| Hard negatives from ranks 10–100 (skip the top 10) | Moreira et al. 2024 (§3.1, §4.3.1); Qu et al. 2021 (§1, §4.2.2) | Faithful |
| Masking other labelled positives out of the denominator | — | Own design (common practice; no single source) |
| Zero-initialised residual bottleneck adapter | Houlsby et al. 2019 (§2.1); Hu et al. 2022 (§4.1); Yoon et al. 2024 (§4.1) | Adapted: query-only adapter; Search-Adaptor shares one adapter for queries and corpus |
| Linear identity-initialised adapter | — | Own design (architecture ablation) |
| Formal preservation $1 - \cos(A_\psi(\mathbf{z}_{\text{form}}), \mathbf{z}_{\text{form}})$ | Taigman et al. 2017 (§4, Eq. 6, $\mathcal{L}_{\text{TID}}$); Zhu et al. 2017 (§5.2, identity loss) | Adapted: identity on target-domain inputs, with cosine distance instead of MSE / L1 |

#### References
* **Ganin & Lempitsky (ICML 2015)**: *Unsupervised Domain Adaptation by Backpropagation*. PMLR 37:1180–1189.
* **Ganin et al. (JMLR 2016)**: *Domain-Adversarial Training of Neural Networks*. JMLR 17(59):1–35.
* **Xin et al. (Findings of ACL 2022)**: *Zero-Shot Dense Retrieval with Momentum Adversarial Domain Invariant Representations* (MoDIR). 4008–4020.
* **van den Oord, Li & Vinyals (2018)**: *Representation Learning with Contrastive Predictive Coding*. arXiv:1807.03748.
* **Karpukhin et al. (EMNLP 2020)**: *Dense Passage Retrieval for Open-Domain Question Answering* (DPR). 6769–6781.
* **Xiong et al. (ICLR 2021)**: *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval* (ANCE).
* **Gao, Yao & Chen (EMNLP 2021)**: *SimCSE: Simple Contrastive Learning of Sentence Embeddings*. 6894–6910.
* **Qu et al. (NAACL 2021)**: *RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering*. 5835–5847.
* **Moreira et al. (2024)**: *NV-Retriever: Improving text embedding models with effective hard-negative mining*. arXiv:2407.15831.
* **Houlsby et al. (ICML 2019)**: *Parameter-Efficient Transfer Learning for NLP*. PMLR 97:2790–2799.
* **Hu et al. (ICLR 2022)**: *LoRA: Low-Rank Adaptation of Large Language Models*.
* **Yoon et al. (ACL 2024)**: *Search-Adaptor: Embedding Customization for Information Retrieval*. 12230–12247.
* **Taigman, Polyak & Wolf (ICLR 2017)**: *Unsupervised Cross-Domain Image Generation*.
* **Zhu et al. (ICCV 2017)**: *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. 2223–2232.
* **Sandriago et al. (ICDEES 2026)**: *ADAPT-DANN: Parameter-Efficient Unsupervised Domain Adaptation for Indonesian Sentiment Analysis* (prior work of this framework; not verifiable online).

---

### Method 2: Query Distillation (Score Distillation)

#### Overview
Uses the frozen base model on formal queries ($\mathbf{z}_{\text{form}}$) as a **Teacher** and trains the informal query adapter $A_\psi(\mathbf{z}_{\text{inf}})$ as a **Student**. Score Distillation matches the soft retrieval probability distribution generated by the teacher across the top $k=100$ candidate corpus passages.

#### Mathematical Formulation

1. **Top-$k$ Candidate Score Distillation Loss ($\tau_{\text{kd}} = 0.05$)**:
   Given teacher top-$k$ retrieved passage candidates $C_k = \{c_1, c_2, \dots, c_{100}\}$:
   $$P_{\text{teacher}} = \text{softmax}\left(\frac{\mathbf{z}_{\text{form}} \cdot C_k}{\tau_{\text{kd}}}\right), \quad P_{\text{student}} = \text{softmax}\left(\frac{A_\psi(\mathbf{z}_{\text{inf}}) \cdot C_k}{\tau_{\text{kd}}}\right)$$
   $$\mathcal{L}_{\text{KD\_Score}} = \text{KL}(P_{\text{teacher}} \parallel P_{\text{student}}) = \sum_{j=1}^k P_{\text{teacher}, j} \log \left(\frac{P_{\text{teacher}, j}}{P_{\text{student}, j}}\right)$$

   $C_k$ is the teacher's top-100 over the **full** 500k corpus, recomputed for every batch.

2. **Total Distillation Loss**:
   $$\mathcal{L}_{\text{Distill}} = \mathcal{L}_{\text{KD\_Score}} + w_{\text{infonce}} \mathcal{L}_{\text{InfoNCE}} + \mu \mathcal{L}_{\text{preserve}}$$
   In the reported runs $w_{\text{infonce}} = 0$ (the default), so the objective is $\mathcal{L}_{\text{KD\_Score}} + 0.5\,\mathcal{L}_{\text{preserve}}$: it uses the informal/formal query pairs but **no relevance labels**. Both adapter variants are reported.

#### Provenance

| Component | Source (section) | Status |
| :--- | :--- | :--- |
| Listwise KL between teacher and student softmax distributions over a candidate list | Ren et al. 2021, RocketQAv2 (§3.2, Eq. 2–4); Zhuang & Zuccon 2022 (§4.2, Eq. 3–4) | Adapted: $\text{KL}(P_{\text{teacher}} \parallel P_{\text{student}})$ (Self-Teaching uses the reverse direction with a stop-gradient teacher); candidates are the teacher's top-100 over the whole corpus instead of a training candidate list |
| Temperature on the distillation softmax ($\tau_{\text{kd}} = 0.05$) | Hinton et al. 2015 | Adapted: the value is this work's choice |
| The clean query teaches its noisy variant with the same model (frozen base on $\mathbf{z}_{\text{form}}$ → adapter on $\mathbf{z}_{\text{inf}}$) | Zhuang & Zuccon 2022, Self-Teaching (§4.2); Sidiropoulos & Kanoulas 2022 | Faithful: typo-free → typoed query there, formal → informal here |
| Adapter and formal preservation term | as Method 1 | as Method 1 |

Hofstätter et al. (SIGIR 2021), cited here originally, is TAS-Balanced: its main distillation loss is pairwise Margin-MSE, and listwise KL appears only in an ablation, so it is background for this loss rather than its source.

#### References
* **Ren et al. (EMNLP 2021)**: *RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking*. 2825–2835.
* **Zhuang & Zuccon (SIGIR 2022)**: *CharacterBERT and Self-Teaching for Improving the Robustness of Dense Retrievers on Queries with Typos*. 1444–1454.
* **Sidiropoulos & Kanoulas (SIGIR 2022)**: *Analysing the Robustness of Dual Encoders for Dense Retrieval Against Misspellings*. 2132–2136.
* **Hinton, Vinyals & Dean (2015)**: *Distilling the Knowledge in a Neural Network*. NIPS 2014 Deep Learning Workshop, arXiv:1503.02531.
* **Hofstätter et al. (SIGIR 2021)**: *Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling*. 113–122 (background).
* **Ryu, Lee & Lee (2022)**: *Knowledge Distillation for BERT Unsupervised Domain Adaptation*. Knowledge and Information Systems; arXiv:2010.11478 (background: adversarial adaptation with classifier distillation).

---

### Method 3: TSDAE-SAPT on Task Queries (Sentence-Adaptive Pre-Training)

#### Overview
Unsupervised full-encoder fine-tuning (`methods/tsdae_sapt_queries/tsdae_train.py`). A Transformer-based Denoising Autoencoder (TSDAE) is trained on the **2,891 informal queries of the train split** (no labels, no dev/test text) to adapt the query encoder. The passage index stays the frozen base index, so the fine-tuned encoder is used for queries only. (A reviews variant, `methods/tsdae_sapt/`, trains on informal product reviews instead; it is not part of the comparison.)

#### Mathematical Formulation
* **Noise Function**: 60% random word deletion on input text $x$.
* **Bottleneck**: The corrupted sentence is passed into the Transformer encoder and pooled to one sentence vector $\mathbf{z}$, with the encoder's retrieval pooling (`[CLS]` for BGE-M3, mean pooling for mE5).
* **Decoder**: A cross-attention Transformer decoder, tied to the encoder's weights, reconstructs the uncorrupted original text from $\mathbf{z}$ only; it is discarded after training.
* **Loss**: Token-level cross-entropy reconstruction loss over sequence vocabulary.
  $$\mathcal{L}_{\text{TSDAE}} = -\sum_{t=1}^T \log P(x_t \mid x_{<t}, \mathbf{z})$$

#### Checkpoint
Reported at its **final step** (1,350), not at the best dev checkpoint like every other trained model: dev selection keeps step 50, which has barely trained and scores like the base model. The final step shows what TSDAE training itself does to retrieval.

*Note on Empirical Findings*: Full encoder fine-tuning via TSDAE degrades dense retrieval (test nDCG@10, informal queries: BGE-M3 0.5362 → 0.4446; mE5 0.5097 → 0.2186; formal queries degrade similarly) because the reconstruction objective is not tied to the retrieval metric space. Freezing the base encoder and training adapters remains superior.

#### Provenance

| Component | Source (section) | Status |
| :--- | :--- | :--- |
| Deletion noise 0.6, `[CLS]` pooling, tied encoder–decoder, reconstruction cross-entropy | Wang, Reimers & Gurevych 2021 (§3.1–3.2, Appendix A) | Faithful: the paper's recommended configuration (mE5 runs use mean pooling, the encoder's own) |
| Adaptive pre-training on the task's own unlabelled text | Gururangan et al. 2020 (TAPT) | Adapted: TSDAE instead of masked language modelling |

#### References
* **Wang, Reimers & Gurevych (Findings of EMNLP 2021)**: *TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning*. 671–688.
* **Gururangan et al. (ACL 2020)**: *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*. 8342–8360.

---

### Method 4: Query InfoNCE (Full Query-Encoder Fine-Tuning)

#### Overview
Supervised full-encoder fine-tuning (`methods/query_infonce/train_infonce.py --objective infonce`). The query encoder is fine-tuned with InfoNCE on **clean informal queries** against the **frozen** passage embeddings of the base index, so retrieval itself pulls informal queries toward the formal-register passages. No adapter, GRL, discriminator or LLM.

#### Mathematical Formulation
* **Query Vector**: $\mathbf{z}_q = \text{L2Norm}(\text{Pool}(\text{Encoder}_\theta(\text{prefix} + x_{\text{inf}})))$ with the encoder's pooling and query prefix (BGE-M3: `[CLS]`, no prefix; mE5: mean pooling, instruction prefix).
* **Loss**: $\mathcal{L}_{\text{InfoNCE}}$ exactly as in Method 1 ($\tau = 0.05$; candidates = the shared pool of $B$ positives + $B$ hard negatives from ranks 10–100, with false-negative masking), where the candidates are rows of the frozen passage index.
* **Objective Variant**: The script's default objective adds a TSDAE reconstruction regularizer ($w_{\text{tsdae}} = 0.1$); the reported model uses `--objective infonce`, i.e. InfoNCE only.

#### Checkpoint
Best dev informal nDCG@10, like the adapter methods.

#### Provenance

| Component | Source (section) | Status |
| :--- | :--- | :--- |
| Fine-tune only the question encoder against a fixed passage index | Sciavolino et al. 2021 (§5, "keeping the passage encoder fixed") | Adapted: trained on informal queries rather than per-relation questions |
| Loss, candidates, hard negatives, $\tau$ | as Method 1 | as Method 1 |

Sciavolino et al. (§4.2) also report that question-only fine-tuning can give small gains when the passage representations are the bottleneck, which is consistent with this method's weak result.

#### References
* **Sciavolino et al. (EMNLP 2021)**: *Simple Entity-Centric Questions Challenge Dense Retrievers*. 6138–6148.
* **Karpukhin et al. (EMNLP 2020)**: *Dense Passage Retrieval for Open-Domain Question Answering* (DPR). 6769–6781.

---

### Method 5: LLM Query Rewrite Baseline (Text-Level Normalization)

#### Overview
Text-level normalization baseline (`methods/llm_rewrite/`): zero-shot LLM rewriting with **gemini-3.5-flash-lite** (an Indonesian instruction prompt, prompt version `id-normalise-v1`; the reported rewrites were requested in batches of 18–40 queries per call, and the batched prompt carries no few-shot examples) transforms informal Indonesian queries into formal sentences, which are then encoded with the base encoder (BGE-M3 or mE5, with that encoder's pooling and prefix). The rewrites are cached, so evaluation makes no API calls. Its formal column rewrites the formal queries too, since a deployed rewriter sees every query. Not trained.

It is a strong **reference, not an upper bound**: it closes most of the informal gap (BGE-M3 +6.21 pts informal nDCG@10, about 80% of the gap), but it also significantly lowers formal-query retrieval (−2.15 pts on BGE-M3, −1.28 on mE5), which the adapter methods do not.

#### Latency (measured on one RTX 3090, BGE-M3, test queries)

| Step | One query at a time | Amortised over a batch |
| :--- | :---: | :---: |
| BGE-M3 query encoding (paid by every pipeline) | 7.43 ms | 0.6 ms (578 queries) |
| Bottleneck adapter forward | 0.14 ms | < 0.004 ms (578 queries) |
| Linear adapter forward | 0.08 ms | < 0.001 ms (578 queries) |
| LLM rewrite, gemini-3.5-flash-lite | not measured | ~55 ms (18–40 queries per API call) |
| LLM rewrite, gemini-2.5-flash | 2.4 s median (41 queries) | — |

The adapter adds ~0.1 ms to the encoder's ~7 ms per query, while the LLM adds a network round trip of tens of milliseconds per query even when batched, and seconds for a single-query call. How measured: the amortised figures come from `methods/llm_rewrite/evaluate_rewrite.py` (LLM latency as cached at rewrite time); the one-at-a-time figures are the mean of 200 single-query encodes and 2,000 single-vector adapter calls after warm-up. A single-query gemini-3.5-flash-lite call was not measured.

#### Provenance

| Component | Source | Status |
| :--- | :--- | :--- |
| Rewrite the query with an LLM before retrieval | Ma et al. 2023 | Adapted: zero-shot prompting of a hosted LLM, no trained rewriter |
| Indonesian colloquial → formal normalisation as a task | Wibowo et al. 2021, IndoCollex | Task definition (word level there, whole query here) |

#### References
* **Ma et al. (EMNLP 2023)**: *Query Rewriting in Retrieval-Augmented Large Language Models*. 5303–5315.
* **Wibowo et al. (Findings of ACL-IJCNLP 2021)**: *IndoCollex: A Testbed for Morphological Transformation of Indonesian Colloquial Words*. 3170–3183.

---

## 4. Hyperparameter Configuration Matrix

### 4.1 Method Hyperparameters

| Parameter | Symbol / Variable | Value | Scope | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Embedding Dimension** | $d$ | `1024` | Global | Output vector size of BGE-M3 & mE5 |
| **Adapter Bottleneck Dim** | $r$ | `128` | Bottleneck Adapter | Inner projection dimension ($1024 \to 128 \to 1024$) |
| **Adapter Dropout** | $p_{\text{drop}}$ | `0.1` | Bottleneck Adapter | Dropout probability inside adapter bottleneck |
| **Discriminator Dropout** | $p_{\text{disc}}$ | `0.2` | Discriminator | Dropout probability inside domain classifier |
| **InfoNCE Temperature** | $\tau$ | `0.05` | Methods 1, 4 | Softmax scaling factor for contrastive loss |
| **Hard-Negative Ranks** | — | `10–100` | Methods 1, 4 | Ranks of the base model's ranking that hard negatives are sampled from |
| **Distillation Temperature**| $\tau_{\text{kd}}$ | `0.05` | Method 2 | Softmax scaling factor for teacher/student scores |
| **Distillation Candidates** | $k$ | `100` | Method 2 | Teacher's top-$k$ passages over the full corpus |
| **Distillation InfoNCE Weight** | $w_{\text{infonce}}$ | `0` | Method 2 | Supervised InfoNCE term switched off (label-free) |
| **Domain Loss Weight** | $\gamma$ | `0.1` | Method 1 | Multiplier for adversarial domain BCE loss |
| **Preservation Loss Weight**| $\mu$ | `0.5` | Methods 1, 2 | Weight for formal vector identity regularizer |
| **GRL Schedule Parameter** | $\eta$ | `10.0` | Method 1 | Exponential growth rate for GRL weight $\alpha_p$ |
| **TSDAE Deletion Ratio** | $r_{\text{del}}$ | `0.6` | Method 3 | Probability of deleting a word in TSDAE noise |

**Note on the mE5 temperature**: $\tau$ and $\tau_{\text{kd}}$ were chosen for BGE-M3. mE5-large-instruct's cosine scores are compressed (relevant and top-ranked non-relevant passages both score ~0.89, random ones ~0.76), so it may be at a disadvantage on Methods 1, 2 and 4. The protocol is kept identical on purpose.

### 4.2 Shared Training Protocol (every reported number)

All trained models (Methods 1–4, both encoders) use one protocol, defined in `comparison/run_all.py` (`PROTOCOL`):

| Setting | Value |
| :--- | :--- |
| **Optimisation Steps** | `1,350` (= 30 epochs × 45 steps of 64 over the 2,891 train queries) |
| **Effective Batch Size** | `64` |
| **Optimiser** | AdamW, lr `3e-5`, weight decay `0.01` |
| **Schedule** | 10% linear warmup, then cosine decay to `1e-6` |
| **Gradient Clipping** | `1.0` |
| **Precision** | bf16 |
| **Max Sequence Length** | `128` (encoder fine-tuning, Methods 3–4) |
| **Seed** | `42`; seed-stability runs repeat every model with seeds `42, 84, 123, 256, 512` |
| **Checkpoint** | Best dev informal nDCG@10, checked every 50 steps (Method 3: final step, see above) |

The trainers' own defaults (e.g. lr `1e-4`, no warmup, no clipping) differ from this protocol; `comparison/run_all.py` passes the protocol explicitly, and no reported number uses the defaults.

---

## 5. Python / PyTorch Implementation Guide

Below are the core PyTorch module definitions for reproducible implementation.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. Zero-Initialized Query Adapter
# ============================================================================
class ZeroInitQueryAdapter(nn.Module):
    """
    Bottleneck query adapter with zero-initialized up-projection.
    Guarantees identity mapping f(x) = x at step 0.
    """
    def __init__(self, in_dim=1024, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.ln = nn.LayerNorm(bottleneck_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, in_dim)
        
        # Zero-initialize up-projection and bias
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        h = self.ln(h)
        h = self.act(h)
        h = self.drop(h)
        delta = self.up(h)
        out = x + delta
        return F.normalize(out, p=2, dim=-1)

class LinearQueryAdapter(nn.Module):
    """
    Adapter variant (--adapter linear): one 1024 x 1024 map initialised to the identity.
    """
    def __init__(self, in_dim=1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=-1)

# ============================================================================
# 2. Gradient Reversal Layer (GRL)
# ============================================================================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)

# ============================================================================
# 3. Sanitized Domain Discriminator
# ============================================================================
class SanitizedDomainDiscriminator(nn.Module):
    """
    Domain classifier with component mean centering to prevent norm shift exploitation.
    """
    def __init__(self, in_dim=1024, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Centering mean component vector
        x_centered = x - x.mean(dim=-1, keepdim=True)
        x_norm = F.normalize(x_centered, p=2, dim=-1)
        return self.net(x_norm)

# ============================================================================
# 4. Multi-Objective Loss Functions
# ============================================================================
def compute_infonce_loss(q_emb, cand_emb, false_neg, tau=0.05):
    """
    q_emb:     (B, 1024) adapted / fine-tuned informal query vectors
    cand_emb:  (2B, 1024) one candidate pool shared by the batch:
               [B positives (row i = query i's target) ; B hard negatives (one per query, ranks 10-100)]
    false_neg: (B, 2B) bool, True where candidate j is a labelled positive of query i but not its own target
    """
    logits = (q_emb.float() @ cand_emb.float().T) / tau          # (B, 2B)
    logits = logits.masked_fill(false_neg, float("-inf"))        # false-negative masking
    labels = torch.arange(q_emb.size(0), device=q_emb.device)    # query i's target is candidate i
    return F.cross_entropy(logits, labels)

def compute_score_kd_loss(student_q, teacher_q, corpus_cand, tau_kd=0.05):
    """
    student_q: (B, 1024)
    teacher_q: (B, 1024)
    corpus_cand: (B, K, 1024) top-k retrieved passages by teacher
    """
    teacher_logits = torch.matmul(teacher_q.unsqueeze(1), corpus_cand.transpose(1, 2)).squeeze(1) / tau_kd
    student_logits = torch.matmul(student_q.unsqueeze(1), corpus_cand.transpose(1, 2)).squeeze(1) / tau_kd
    
    p_teacher = F.softmax(teacher_logits, dim=-1)
    log_p_student = F.log_softmax(student_logits, dim=-1)
    
    return F.kl_div(log_p_student, p_teacher, reduction='batchmean')

def compute_formal_preservation_loss(adapter, formal_q_emb):
    """
    Ensures adapter(z_formal) ≈ z_formal
    """
    adapted_formal = adapter(formal_q_emb)
    cosine_sim = torch.sum(adapted_formal * formal_q_emb, dim=-1)
    return torch.mean(1.0 - cosine_sim)
```

---

## 6. Primary Evaluation Metrics & Benchmark Protocols

Every model of an encoder is scored on the same 553 test queries (formal and informal version) against that encoder's frozen 500k index, with binary relevance restricted to passages present in the index.

1. **nDCG@10 (Normalized Discounted Cumulative Gain at Rank 10)**: Primary ranking metric evaluating precision and top-rank relevance quality.
2. **MRR@10 (Mean Reciprocal Rank at Rank 10)**: Evaluates the rank position of the first relevant retrieved document.
3. **Recall@100**: Evaluates global candidate retrieval coverage across the top 100 passages.
4. **Gap Closed**: Share of the base model's formal–informal nDCG@10 gap recovered on informal queries, $100 \cdot (\text{nDCG}_{\text{inf}}^{\text{model}} - \text{nDCG}_{\text{inf}}^{\text{base}}) / (\text{nDCG}_{\text{form}}^{\text{base}} - \text{nDCG}_{\text{inf}}^{\text{base}})$.
5. **Significance Testing**: Paired bootstrap over queries against the same encoder's base model ($N = 1000$ resamples, seed 0). The 95% confidence interval is the percentile interval of the resampled mean difference; the two-sided $p$-value comes from the null-shifted bootstrap (differences re-centred to mean 0) with the $(\text{count} + 1) / (N + 1)$ correction, so $p$ is never reported as exactly 0.
6. **Seed Stability**: Every trained model is repeated with seeds 42, 84, 123, 256 and 512 (`comparison/run_all.py --std true`). Reported per model: mean, sample standard deviation across seeds ($n - 1$, in pts), the per-seed scores and selected checkpoint steps, and how many seeds are significantly better / worse than base on their own (paired bootstrap, $p < 0.05$).
7. **Encoder Comparison**: The same models are trained and scored on both encoders (`--encoder compare`); every gain is measured against that encoder's own base model.

**Provenance**: nDCG follows Järvelin & Kekäläinen (2002) with binary gains; MRR follows the TREC-8 QA track (Voorhees 1999), cut at rank 10. The significance test is the bootstrap "shift method" of Smucker, Allan & Carterette (2007, §2.4), with the $(r+1)/(n+1)$ p-value of North, Curtis & Sham (2002); it uses 1,000 resamples where Smucker et al. use 100,000, so the smallest reportable $p$ is 0.001. All four are checked against independent implementations in `audit/crosscheck.py` (checks 8–9).

**Reports** (plain-text tables in `comparison/outputs/`):

| File | Contents |
| :--- | :--- |
| `report_test.md`, `report_test_me5.md` | All metrics, gains, gap closed, 95% CIs, $p$-values and checkpoint steps per encoder (seed 42) |
| `report_encoders_test.md` | BGE-M3 and mE5 side by side |
| `report_std_test.md`, `report_std_test_me5.md`, `report_encoders_std_test.md` | Seed stability: mean, std and per-seed results |

---

## 7. Official Literature & Method References

Every entry was checked against its publisher, ACL Anthology, PMLR or arXiv record on 2026-10-19. Entries marked † were wrong in the earlier version of this spec and are corrected here. Su et al. (ACL 2022) was removed: the paper is real, but it compares active learning, self-training and data augmentation for source-free domain adaptation and contains no preservation regularizer.

1. **Chen, Xiao, Zhang, Luo, Lian & Liu (2024)**: M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. *Findings of ACL 2024*, 2318–2335.
2. **Ganin & Lempitsky (2015)**: Unsupervised Domain Adaptation by Backpropagation. *Proc. ICML 2015*, PMLR 37:1180–1189.
3. **Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand & Lempitsky (2016)**: Domain-Adversarial Training of Neural Networks. *Journal of Machine Learning Research*, 17(59), 1–35.
4. **Gao, Yao & Chen (2021)**: SimCSE: Simple Contrastive Learning of Sentence Embeddings. *Proc. EMNLP 2021*, 6894–6910.
5. **Gururangan, Marasović, Swayamdipta, Lo, Beltagy, Downey & Smith (2020)**: Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. *Proc. ACL 2020*, 8342–8360.
6. **Hinton, Vinyals & Dean (2015)**: Distilling the Knowledge in a Neural Network. *NIPS 2014 Deep Learning Workshop*. arXiv:1503.02531.
7. **Hofstätter, Lin, Yang, Lin & Hanbury (2021)** †: Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling. *Proc. SIGIR 2021*, 113–122.
8. **Houlsby, Giurgiu, Jastrzebski, Morrone, de Laroussilhe, Gesmundo, Attariyan & Gelly (2019)**: Parameter-Efficient Transfer Learning for NLP. *Proc. ICML 2019*, PMLR 97:2790–2799.
9. **Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen (2022)**: LoRA: Low-Rank Adaptation of Large Language Models. *Proc. ICLR 2022*. arXiv:2106.09685.
10. **Järvelin & Kekäläinen (2002)**: Cumulated Gain-Based Evaluation of IR Techniques. *ACM Transactions on Information Systems*, 20(4), 422–446.
11. **Karpukhin, Oguz, Min, Lewis, Wu, Edunov, Chen & Yih (2020)** †: Dense Passage Retrieval for Open-Domain Question Answering. *Proc. EMNLP 2020*, 6769–6781.
12. **Koto, Lau & Baldwin (2021)**: IndoBERTweet: A Pretrained Language Model for Indonesian Twitter with Effective Domain-Specific Vocabulary Initialization. *Proc. EMNLP 2021*, 10660–10668.
13. **Ma, Gong, He, Zhao & Duan (2023)**: Query Rewriting in Retrieval-Augmented Large Language Models. *Proc. EMNLP 2023*, 5303–5315.
14. **Moreira, Osmulski, Xu, Ak, Schifferer & Oldridge (2024)**: NV-Retriever: Improving Text Embedding Models with Effective Hard-Negative Mining. arXiv:2407.15831.
15. **North, Curtis & Sham (2002)**: A Note on the Calculation of Empirical P Values from Monte Carlo Procedures. *American Journal of Human Genetics*, 71(2), 439–441.
16. **Pfeiffer, Rücklé, Poth, Kamath, Vulić, Ruder, Cho & Gurevych (2020)**: AdapterHub: A Framework for Adapting Transformers. *Proc. EMNLP 2020 System Demonstrations*, 46–54.
17. **Qu, Ding, Liu, Liu, Ren, Zhao, Dong, Wu & Wang (2021)**: RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. *Proc. NAACL 2021*, 5835–5847.
18. **Reimers & Gurevych (2020)**: Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation. *Proc. EMNLP 2020*. arXiv:2004.09813 (the method behind paraphrase-multilingual-mpnet-base-v2).
19. **Ren, Qu, Liu, Zhao, She, Wu, Wang & Wen (2021)**: RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking. *Proc. EMNLP 2021*, 2825–2835.
20. **Ryu, Lee & Lee (2022)**: Knowledge Distillation for BERT Unsupervised Domain Adaptation. *Knowledge and Information Systems*. arXiv:2010.11478.
21. **Sandriago et al. (2026)**: ADAPT-DANN: Parameter-Efficient Unsupervised Domain Adaptation for Indonesian Sentiment Analysis. *Proc. ICDEES 2026* (not verifiable online).
22. **Sciavolino, Zhong, Lee & Chen (2021)**: Simple Entity-Centric Questions Challenge Dense Retrievers. *Proc. EMNLP 2021*, 6138–6148.
23. **Sidiropoulos & Kanoulas (2022)**: Analysing the Robustness of Dual Encoders for Dense Retrieval Against Misspellings. *Proc. SIGIR 2022*, 2132–2136.
24. **Smucker, Allan & Carterette (2007)**: A Comparison of Statistical Significance Tests for Information Retrieval Evaluation. *Proc. CIKM 2007*, 623–632.
25. **Taigman, Polyak & Wolf (2017)**: Unsupervised Cross-Domain Image Generation. *Proc. ICLR 2017*. arXiv:1611.02200.
26. **van den Oord, Li & Vinyals (2018)**: Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748.
27. **Voorhees (1999)**: The TREC-8 Question Answering Track Report. *Proc. TREC-8*, NIST Special Publication 500-246.
28. **Wang, Reimers & Gurevych (2021)** †: TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning. *Findings of EMNLP 2021*, 671–688.
29. **Wang, Yang, Huang, Yang, Majumder & Wei (2024)**: Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672.
30. **Wibowo, Nityasya, Akyürek, Fitriany, Aji, Prasojo & Wijaya (2021)**: IndoCollex: A Testbed for Morphological Transformation of Indonesian Colloquial Words. *Findings of ACL-IJCNLP 2021*, 3170–3183.
31. **Xin, Xiong, Srinivasan, Sharma, Jose & Bennett (2022)**: Zero-Shot Dense Retrieval with Momentum Adversarial Domain Invariant Representations. *Findings of ACL 2022*, 4008–4020.
32. **Xiong, Xiong, Li, Tang, Liu, Bennett, Ahmed & Overwijk (2021)**: Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. *Proc. ICLR 2021*.
33. **Yoon, Chen, Arik & Pfister (2024)**: Search-Adaptor: Embedding Customization for Information Retrieval. *Proc. ACL 2024*, 12230–12247.
34. **Zhang, Thakur, Ogundepo, Kamalloo, Alfonso-Hermelo, Li, Liu, Rezagholizadeh & Lin (2023)**: MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages. *Transactions of the ACL*, 11, 1114–1131.
35. **Zhu, Park, Isola & Efros (2017)**: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. *Proc. ICCV 2017*, 2223–2232.
36. **Zhuang & Zuccon (2022)**: CharacterBERT and Self-Teaching for Improving the Robustness of Dense Retrievers on Queries with Typos. *Proc. SIGIR 2022*, 1444–1454.
