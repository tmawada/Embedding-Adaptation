# Credibility audit: training pipeline vs published research and the spec

- Scope:     Query-DANN, query distillation, TSDAE-SAPT, Query InfoNCE, LLM rewrite, data, evaluation
- Sources:   every reference below was opened and checked on 2026-10-19
- Code:      the checks import and run the project's own code on the real cached data
- Rerun:     python3 audit/crosscheck.py

## 1. Verdict

| Provenance verdict | Components | Ids                                         |
|--------------------|-----------:|---------------------------------------------|
| faithful           |         10 | P1, P2, P6, P8, P9, P15, P16, P20, P21, P22 |
| adapted            |          9 | P3, P4, P7, P11, P13, P14, P17, P18, P19    |
| own design         |          3 | P5, P10, P12                                |

Runnable checks: 13 of 13 pass.

faithful = the component is the published method with the published settings; adapted = taken from
the cited work with the stated changes; own design = no published source was found, the component is
this project's choice (said so explicitly rather than given a citation it does not have).

## 2. Research provenance

| Id  | Component                                            | Verdict    | Sources                       |
|-----|------------------------------------------------------|------------|-------------------------------|
| P1  | Gradient reversal layer                              | faithful   | ganin15, ganin16              |
| P2  | GRL schedule alpha_p = 2/(1+exp(-10p)) - 1           | faithful   | ganin15, ganin16              |
| P3  | Domain loss: binary cross-entropy, informal -> 0,... | adapted    | ganin16                       |
| P4  | Adversarial domain adaptation for dense retrieval    | adapted    | modir                         |
| P5  | Sanitized discriminator (centre + re-normalise in... | own design | -                             |
| P6  | InfoNCE / negative log-likelihood of the positive    | faithful   | cpc, dpr                      |
| P7  | Candidates: in-batch positives + one hard negativ... | adapted    | dpr, ance                     |
| P8  | Temperature tau = 0.05 on cosine similarity          | faithful   | simcse                        |
| P9  | Hard negatives from ranks 10-100, labelled positi... | faithful   | nvret, rocketqa               |
| P10 | False-negative masking of other labelled positive... | own design | -                             |
| P11 | Zero-initialised residual bottleneck adapter (102... | adapted    | houlsby, lora, searchadaptor  |
| P12 | Linear adapter, 1024 x 1024 initialised to the id... | own design | -                             |
| P13 | Formal preservation regulariser mean(1 - cos(A(z_... | adapted    | taigman, cyclegan             |
| P14 | Listwise KL score distillation over the teacher's... | adapted    | rocketqav2, selfteach, hinton |
| P15 | Teacher = frozen base model on the formal query, ... | faithful   | selfteach, misspell           |
| P16 | TSDAE: 60% word deletion, CLS pooling, tied encod... | faithful   | tsdae                         |
| P17 | Adaptive pre-training on the task's own unlabelle... | adapted    | dapt                          |
| P18 | Fine-tune only the query encoder against a fixed ... | adapted    | entityq, dpr                  |
| P19 | Rewrite the informal query with an LLM, then retr... | adapted    | qrewrite, indocollex          |
| P20 | nDCG@10, MRR@10, Recall@100 with binary relevance    | faithful   | ndcg, mrr                     |
| P21 | Paired bootstrap test, shift method, two-sided       | faithful   | smucker, north                |
| P22 | MIRACL-id, BGE-M3, mE5-large-instruct, multilingu... | faithful   | miracl, bgem3, me5, mpnet     |

P1. Gradient reversal layer (Query-DANN; methods/model.py:87) - faithful.
    Source: ICML 2015 Sec. 3; JMLR 2016 Sec. 4.2.
    Forward identity, backward gradient multiplied by -lambda, as in Ganin & Lempitsky 2015 (the
    JMLR version reverses with -1 and puts lambda in the loss; the two are equivalent for the
    feature extractor).

P2. GRL schedule alpha_p = 2/(1+exp(-10p)) - 1 (Query-DANN; methods/model.py:108) - faithful.
    Source: JMLR 2016 Sec. 5.2.2 (gamma = 10 in all experiments).
    Same formula and gamma = 10, p = step / total steps (the step counter of the optimiser).

P3. Domain loss: binary cross-entropy, informal -> 0, formal -> 1, weight gamma = 0.1 (Query-DANN;
    methods/query_dann/train.py:190, :193) - adapted.
    Source: JMLR 2016 Sec. 4.1 (binomial cross-entropy).
    Logistic / binary cross-entropy on discriminator logits, the two domains weighted 0.5 / 0.5.
    Difference: the loss is added with weight gamma = 0.1, so the adapter's adversarial weight is
    gamma x alpha_p <= 0.1 where DANN uses alpha_p. The discriminator's gradient is scaled by 0.1 as
    well, which AdamW largely normalises away (the discriminator gets no other gradient).

P4. Adversarial domain adaptation for dense retrieval (Query-DANN; methods/query_dann/train.py) -
    adapted.
    Source: Findings ACL 2022 (domain classifier + adversarial update of the DR encoder).
    MoDIR aligns source and target corpora by adversarially training the whole encoder (with a
    momentum classifier). Here the domains are informal vs formal queries and only a query adapter
    on a frozen encoder is trained.

P5. Sanitized discriminator (centre + re-normalise input) (Query-DANN; methods/model.py:201) - own
    design.
    Source: none found.
    No published source found. Introduced in this project because a LayerNorm adapter output gave
    the discriminator a trivial 100%-accurate shortcut (docstring of ZeroInitQueryAdapter).

P6. InfoNCE / negative log-likelihood of the positive (Query-DANN, InfoNCE;
    methods/query_dann/train.py:187) - faithful.
    Source: CPC Sec. 2.3 Eq. 4; DPR Sec. 3.2 Eq. 2.
    Cross-entropy of the positive among the candidates, cosine similarity / tau.

P7. Candidates: in-batch positives + one hard negative per query, shared by the batch (Query-DANN,
    InfoNCE; preprocessing/dataset.py:141) - adapted.
    Source: DPR Sec. 3.2 and 5.2; ANCE (model-mined negatives).
    DPR's best setting adds one BM25 negative per question and uses it as a negative for all
    questions in the batch. Same structure here, but the negative is mined with the dense base model
    (ANCE-style) instead of BM25.

P8. Temperature tau = 0.05 on cosine similarity (Query-DANN, InfoNCE;
    methods/query_dann/train.py:184) - faithful.
    Source: SimCSE Appendix D (tau = 0.05 with cosine similarity).
    Same value and similarity.

P9. Hard negatives from ranks 10-100, labelled positives removed (Query-DANN, InfoNCE;
    methods/prepare.py:89, preprocessing/dataset.py:123) - faithful.
    Source: NV-Retriever Sec. 3.1 ('Top-K shifted by N') and 4.3.1; RocketQA Sec. 1 and 4.2.2.
    Skipping the top-ranked candidates because they often are unlabelled positives is NV-Retriever's
    'shifted by N' mining (their best shifted setting discards the top 10); RocketQA reports ~70% of
    unlabelled top-retrieved passages are relevant.

P10. False-negative masking of other labelled positives in the denominator (Query-DANN, InfoNCE;
    preprocessing/dataset.py:145) - own design.
    Source: none found.
    Common practice in contrastive retrieval training code; no single peer-reviewed source verified.
    Motivated by the false-negative problem documented in RocketQA.

P11. Zero-initialised residual bottleneck adapter (1024 -> 128 -> 1024) (Query-DANN, Distillation;
    methods/model.py:152) - adapted.
    Source: Houlsby Sec. 2-2.1; LoRA Sec. 4.1; Search-Adaptor Sec. 4.1.
    Bottleneck with internal skip connection and (near-)identity initialisation from Houlsby et al.;
    exact identity start by zero-initialising the up-projection as LoRA initialises B = 0; a
    residual adapter on top of frozen embeddings for retrieval as in Search-Adaptor. Differences:
    Search-Adaptor shares one adapter between queries and corpus, here only queries are adapted and
    the index is untouched.

P12. Linear adapter, 1024 x 1024 initialised to the identity (Query-DANN, Distillation;
    methods/model.py:225) - own design.
    Source: none found.
    Ablation of the adapter architecture; not in the original spec. No source claimed.

P13. Formal preservation regulariser mean(1 - cos(A(z_form), z_form)) (Query-DANN, Distillation;
    methods/query_dann/train.py:191) - adapted.
    Source: Taigman et al. Sec. 4 Eq. 6 (L_TID); CycleGAN Sec. 5.2 (identity mapping loss).
    Both regularise the mapping to be the identity on target-domain inputs (formal queries here).
    They use MSE / L1; here the distance is 1 - cosine. The spec cites Su et al. (ACL 2022) for this
    term, which does not contain it (see Citations).

P14. Listwise KL score distillation over the teacher's top-k, with temperature (Distillation;
    methods/query_distillation/train.py:237-249) - adapted.
    Source: RocketQAv2 Sec. 3.2 Eq. 2-4; Self-Teaching Sec. 4.2 Eq. 3-4; Hinton et al.
    (temperature).
    Softmax over a candidate list for teacher and student, KL between them. Differences: KL(teacher
    || student) (Hinton's direction; Self-Teaching uses KL(student || stop-grad teacher));
    candidates are the teacher's top-100 over the full corpus instead of a training candidate list;
    temperature 0.05. The spec cites Hofstaetter et al. for this loss, but that paper's main loss is
    Margin-MSE (KL appears only in an ablation).

P15. Teacher = frozen base model on the formal query, student = adapted informal query
    (Distillation; methods/query_distillation/train.py:236) - faithful.
    Source: Self-Teaching Sec. 4.2 ('the model itself is both the student and the teacher').
    Clean query teaches its noisy variant with the same model, as in Self-Teaching (typo ->
    informal). Sidiropoulos & Kanoulas pull typoed queries toward their clean version for the same
    goal.

P16. TSDAE: 60% word deletion, CLS pooling, tied encoder-decoder, reconstruction CE (TSDAE-SAPT;
    methods/tsdae_sapt_queries/tsdae_train.py:41, :71, :143) - faithful.
    Source: TSDAE Sec. 3.1-3.2 and Appendix A.
    Deletion ratio 0.6, [CLS] pooling and tied weights are the paper's recommended configuration
    (mE5 runs use mean pooling, the encoder's own pooling).

P17. Adaptive pre-training on the task's own unlabelled text (TSDAE-SAPT;
    methods/tsdae_sapt_queries/tsdae_train.py:138) - adapted.
    Source: Gururangan et al. (TAPT).
    Task-adaptive pre-training on the task's unlabelled text (here: the train-split informal
    queries), with TSDAE instead of masked language modelling.

P18. Fine-tune only the query encoder against a fixed passage index (Query InfoNCE;
    methods/query_infonce/train_infonce.py:296) - adapted.
    Source: Sciavolino et al. Sec. 5 ('keeping the passage encoder fixed'); DPR loss.
    Same setting as Sciavolino et al.'s specialised question encoders, trained with the DPR /
    InfoNCE loss on informal queries. Not in the original spec.

P19. Rewrite the informal query with an LLM, then retrieve with the base encoder (LLM rewrite;
    methods/llm_rewrite/normalize.py) - adapted.
    Source: Ma et al. (LLM query rewriting); IndoCollex (Indonesian colloquial -> formal).
    Zero-shot instruction prompt (gemini-3.5-flash-lite), no trained rewriter; the task (colloquial
    -> formal Indonesian) is the one IndoCollex studies at the word level.

P20. nDCG@10, MRR@10, Recall@100 with binary relevance (Evaluation; methods/eval.py:45) - faithful.
    Source: Jarvelin & Kekalainen; TREC-8 QA (reciprocal rank).
    Log2 discount, ideal DCG over min(|rel|, 10) relevant passages; reciprocal rank of the first hit
    in the top 10.

P21. Paired bootstrap test, shift method, two-sided (Evaluation; methods/eval.py:69) - faithful.
    Source: Smucker et al. Sec. 2.4; North et al. (r+1)/(n+1).
    Resample query pairs, shift the bootstrap distribution to mean 0, two-sided p = share with
    |diff| >= observed, with the (r+1)/(n+1) correction. 1,000 resamples (Smucker et al. use
    100,000), so the smallest p is 0.001.

P22. MIRACL-id, BGE-M3, mE5-large-instruct, multilingual-mpnet (Data / encoders; methods/prepare.py,
    methods/encoders.py) - faithful.
    Source: -.
    Public dataset and checkpoints used as released. BGE-M3 and mE5 list MIRACL training data in
    their fine-tuning (see Caveats); multilingual-mpnet (2020, distilled on parallel data) predates
    MIRACL.

## 3. Runnable checks

| Check | Result | Name                                                                     |
|-------|--------|--------------------------------------------------------------------------|
| 1     | PASS   | Adapters start as the identity; parameter counts                         |
| 2     | PASS   | Gradient reversal: identity forward, -alpha x gradient backward; alph... |
| 3     | PASS   | Sanitized discriminator ignores a constant shift and the scale of its... |
| 4     | PASS   | Query-DANN train_step losses equal independent per-query reference lo... |
| 5     | PASS   | Batch construction: negatives from ranks 10-100, never positives; mas... |
| 6     | PASS   | Distillation: KL expression equals sum p_t log(p_t / p_s) / B; teache... |
| 7a    | PASS   | TSDAE noise deletes ~60% of words                                        |
| 7b    | PASS   | TSDAE decoder is tied to the encoder; CLS pooling                        |
| 8     | PASS   | Metrics equal an independent implementation, scikit-learn's ndcg_scor... |
| 9     | PASS   | Paired bootstrap: shift method, p = 1 for identical systems, ~5% fals... |
| 10    | PASS   | No leakage: disjoint splits, no repeated questions in dev / test, tra... |
| 11    | PASS   | Cached query and passage vectors are reproduced by every encoder (BGE... |
| 12    | PASS   | Reported numbers are reproduced from the saved checkpoints               |

1. Adapters start as the identity; parameter counts [methods/model.py:152, :225]: PASS.
    max |A(z) - z| = 3.0e-08 in train and eval mode; parameters bottleneck 263,552, linear 1,049,600

2. Gradient reversal: identity forward, -alpha x gradient backward; alpha_p schedule
    [methods/model.py:87, :108]: PASS.
    forward error 0.0e+00, backward error 0.0e+00 (alpha 0 / 0.3 / 1), schedule error 0.0e+00;
    alpha(0) = 0.000, alpha(1) = 0.99991

3. Sanitized discriminator ignores a constant shift and the scale of its input
    [methods/model.py:201]: PASS.
    max logit change 3.4e-08 under z + c and 3z (the same network without centring changes by 0.03,
    so the centring is what removes the shortcut)

4. Query-DANN train_step losses equal independent per-query reference losses
    [methods/query_dann/train.py:172]: PASS.
    code vs reference: InfoNCE 0.41002 / 0.41002, domain 0.15044 / 0.15044, preserve 0.032004 /
    0.032004; max abs diff 1.2e-08 (trained seed-42 checkpoint, 64 train queries, fp32)

5. Batch construction: negatives from ranks 10-100, never positives; masking rule
    [preprocessing/dataset.py:111-160, methods/prepare.py:89]: PASS.
    labelled positives in mined negative lists: 0; over one epoch (2891 queries) negatives outside
    ranks 10-100: 0, negatives that are positives: 0; masking rule mismatches over 30 batches: 0 (18
    entries legitimately masked)

6. Distillation: KL expression equals sum p_t log(p_t / p_s) / B; teacher candidates are its top-100
    [methods/query_distillation/train.py:237-249]: PASS.
    the 4 loss expressions are present in the code: True; KL code 0.028164 vs manual 0.028164 (diff
    2.4e-09); teacher top-100 in fp16 vs exact fp32 overlap 99.58%

7a. TSDAE noise deletes ~60% of words [methods/tsdae_sapt_queries/tsdae_train.py:41]: PASS.
    58.9% of 16,962 words deleted over the 2891 train informal queries (slightly below 60% because
    one word is always kept)

7b. TSDAE decoder is tied to the encoder; CLS pooling [methods/tsdae_sapt_queries/tsdae_train.py:71,
    :143]: PASS.
    389 of 634 decoder parameter tensors are the encoder's own tensors (the rest are the new
    cross-attention and LM head); pooling = CLS: True

8. Metrics equal an independent implementation, scikit-learn's ndcg_score and hand-computed cases
    [methods/eval.py:45]: PASS.
    hand cases max error 0.0e+00; 500 random queries vs independent code 0.0e+00, vs sklearn
    ndcg_score 1.4e-16

9. Paired bootstrap: shift method, p = 1 for identical systems, ~5% false positives under the null
    [methods/eval.py:69]: PASS.
    identical systems p = 1.000; clear shift p = 0.0010; vs independent shift-method code diff
    0.0e+00; false-positive rate at p < 0.05 over 300 null pairs 5.0%

10. No leakage: disjoint splits, no repeated questions in dev / test, trainers read only train
    [methods/cache/splits.json, preprocessing/dataset.py drop_twins, the four trainers,
    normalize.py]: PASS.
    split overlaps 0; train + dev + test + 39 excluded twins = the 3,855 answerable queries: True;
    sizes 2891 / 372 / 553; training queries outside train: 0; references to the test split in the
    trainers: 0; TSDAE text = train informal queries: True; LLM few-shot pool = train: True; dev /
    test queries repeating a train question: 0 / 0; repeats among dev + test: 0

11. Cached query and passage vectors are reproduced by every encoder (BGE-M3 CLS; mE5, mpnet mean)
    [methods/model.py:20, methods/encoders.py]: PASS.
    BGE-M3: min cosine formal 0.99993, informal 0.99996, passages 0.99991; mE5-large-instruct: min
    cosine formal 0.99994, informal 0.99993, passages 0.99994; multilingual-mpnet: min cosine formal
    0.99996, informal 0.99996, passages 0.99994 (100 test queries, 50 passages)

12. Reported numbers are reproduced from the saved checkpoints [comparison/evaluate_all.py]: PASS.
    report_test.json identical: True; report_test_me5.json identical: True; report_test_mpnet.json
    identical: True

## 4. Code vs the original spec

Spec as originally given (archived as
../bge_dann_archive/informal-query-embedding-adaptation-spec.orig.md). Everything not listed
matches: adapter, GRL and its schedule, gamma 0.1, mu 0.5, eta 10, tau 0.05, hard negatives 10-100
with masking (formula text), score KD (k 100, tau_kd 0.05), TSDAE deletion 0.6, metrics and
bootstrap.

- Spec: mE5 = multilingual-e5-large with 'query: ' / 'passage: '
  Code: multilingual-e5-large-instruct, instruction prefix on queries, no passage prefix
  (methods/encoders.py)
- Spec: 2,891 query pairs split 75/10/15
  Code: 3,855 pairs split 2,891 / 386 / 578, then 39 dev / test queries that repeat an earlier
  question dropped: 2,891 / 372 / 553 (methods/cache/splits.json, twins_excluded)
- Spec: 30 epochs = ~1,020 steps
  Code: 30 x 45 steps = 1,350 steps (logs/query_dann.log)
- Spec: lr 1e-4, no warmup or clipping
  Code: Reported runs: lr 3e-5 -> 1e-6, 10% warmup, clip 1.0 (comparison/run_all.py)
- Spec: Discriminator formula 1024 -> 512 -> 1
  Code: 1024 -> 512 -> 256 -> 1 (methods/model.py:178; the spec's own code block agrees with the
  code)
- Spec: Reference InfoNCE code: per-query negatives, no masking
  Code: Shared 2B candidate pool with false-negative masking (the spec's formula text says masking;
  its code does not)
- Spec: LLM ~500 ms, adapter <1 ms, ~1000x
  Code: Not in the code; measured: encoder 7.43 ms, bottleneck adapter 0.14 ms, flash-lite ~55
  ms/query batched
- Spec: LLM = 'Gemini Flash', accuracy upper bound
  Code: gemini-3.5-flash-lite; lowers formal nDCG@10 by 2.00 pts, so not an upper bound
- Spec: TSDAE on informal text (reviews or queries)
  Code: Reported model: train-split informal queries, scored at its final step
- Spec: Methods 4-5: Stacked, UDA Query-DANN
  Code: Code exists (methods/stacked, methods/uda_query_dann) but both were dropped from the
  comparison; not audited
- Spec: (not in spec)
  Code: Query InfoNCE (full query-encoder fine-tuning) and the linear adapter are trained and
  reported

## 5. Citations in the original spec

- Hofstaetter et al. (SIGIR 2021), 'Efficiently Teaching an Ensemble of Dense Retrievers with a
  Unified Knowledge Distillation Technique', 1414-1423
  -> Title and pages are wrong: 'Efficiently Teaching an Effective Dense Retriever with Balanced
  Topic Aware Sampling', 113-122. Its main loss is Margin-MSE (KL only in an ablation), so it is not
  the source of the listwise KL used here; cite RocketQAv2 / Self-Teaching instead.
- Su et al. (ACL 2022), cited for the formal preservation regulariser
  -> Real paper (8352-8367), but it compares active learning, self-training and data augmentation
  for source-free domain adaptation; it has no identity / preservation loss. Use Taigman et al. 2017
  / CycleGAN.
- Reimers & Gurevych (EMNLP 2021), TSDAE, 2129-2141
  -> Authors, venue and pages are wrong: Wang, Reimers & Gurevych, Findings of EMNLP 2021, 671-688.
- Karpukhin et al. (EMNLP 2020), DPR, 4171-4186
  -> Pages are wrong (4171-4186 are BERT's NAACL 2019 pages): 6769-6781.
- Ryu et al. (2022), Knowledge Distillation for BERT UDA
  -> Real (arXiv:2010.11478; journal version Knowledge and Information Systems,
  doi:10.1007/s10115-022-01736-y), but it combines adversarial adaptation with classifier
  distillation for sentiment; background only, not the source of the score-distillation loss.
- Ganin et al. (JMLR 2016); Gururangan et al. (ACL 2020); Pfeiffer et al. (EMNLP 2020 demos); Koto
  et al. (EMNLP 2021)
  -> Correct as cited.
- Sandriago et al. (ICDEES 2026)
  -> Not verifiable online (the author's own paper); not checked.

## 6. Caveats

- Contamination: all queries come from the official MIRACL-id train split, and both encoders list
  MIRACL / Mr. TyDi training data in their fine-tuning, so formal scores are probably inflated.
  Method comparisons share the exposure; absolute numbers are not comparable to published MIRACL
  results.
- Duplicate questions (resolved 2026-10-19): MIRACL-id train repeats some questions under different
  ids. The 39 dev / test queries that repeated an earlier question (14 dev, 25 test) were dropped
  (methods/drop_twins.py; ids in splits.json twins_excluded; the old split is kept as
  splits_with_twins.json) and every model was retrained. Before that, re-scoring without the 24 test
  twins had dropped Query-DANN (bottleneck) on BGE-M3 from +1.76 to +1.27 pts (no longer
  significant) while the distillation gains barely moved.
- Single seed (42) for the reported numbers. The five-seed run was stopped before it finished, and
  its seed runs were selected on the dev split that still contained repeated questions: a later
  --std true run needs --force.
- tau = 0.05 was chosen for BGE-M3; mE5's compressed cosine scores may put it at a disadvantage.
- Checkpoints are selected on dev informal nDCG@10 (dev labels); TSDAE is reported at its final
  step, a choice made after seeing that dev selection keeps step 50.
- Training runs adapter / discriminator forward passes under bf16 autocast; check 4 runs the same
  code in fp32 so the reference comparison is exact. The distillation teacher scores the corpus in
  fp16 (check 6 measures the effect on its top-100).
- normalize.py records 'shots': 3 for every cached rewrite, but the batched requests used for the
  reported rewrites carry no few-shot examples (do_batch builds the prompt from INSTRUCTION only):
  the LLM row is zero-shot.

## References (verified)

[ganin15] Ganin & Lempitsky (2015). Unsupervised Domain Adaptation by Backpropagation. ICML 2015,
    PMLR 37:1180-1189. https://proceedings.mlr.press/v37/ganin15.html
[ganin16] Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand & Lempitsky (2016).
    Domain-Adversarial Training of Neural Networks. JMLR 17(59):1-35.
    https://jmlr.org/papers/v17/15-239.html
[modir] Xin, Xiong, Srinivasan, Sharma, Jose & Bennett (2022). Zero-Shot Dense Retrieval with
    Momentum Adversarial Domain Invariant Representations. Findings of ACL 2022, 4008-4020.
    https://aclanthology.org/2022.findings-acl.316/
[cpc] van den Oord, Li & Vinyals (2018). Representation Learning with Contrastive Predictive Coding.
    arXiv:1807.03748.
[dpr] Karpukhin, Oguz, Min, Lewis, Wu, Edunov, Chen & Yih (2020). Dense Passage Retrieval for
    Open-Domain Question Answering. EMNLP 2020, 6769-6781.
    https://aclanthology.org/2020.emnlp-main.550/
[simcse] Gao, Yao & Chen (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP
    2021, 6894-6910. https://aclanthology.org/2021.emnlp-main.552/
[ance] Xiong, Xiong, Li, Tang, Liu, Bennett, Ahmed & Overwijk (2021). Approximate Nearest Neighbor
    Negative Contrastive Learning for Dense Text Retrieval. ICLR 2021. arXiv:2007.00808.
[rocketqa] Qu, Ding, Liu, Liu, Ren, Zhao, Dong, Wu & Wang (2021). RocketQA: An Optimized Training
    Approach to Dense Passage Retrieval for Open-Domain Question Answering. NAACL 2021, 5835-5847.
    https://aclanthology.org/2021.naacl-main.466/
[nvret] Moreira, Osmulski, Xu, Ak, Schifferer & Oldridge (2024). NV-Retriever: Improving text
    embedding models with effective hard-negative mining. arXiv:2407.15831.
[houlsby] Houlsby, Giurgiu, Jastrzebski, Morrone, de Laroussilhe, Gesmundo, Attariyan & Gelly
    (2019). Parameter-Efficient Transfer Learning for NLP. ICML 2019, PMLR 97:2790-2799.
[lora] Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen (2022). LoRA: Low-Rank Adaptation of Large
    Language Models. ICLR 2022. arXiv:2106.09685.
[searchadaptor] Yoon, Chen, Arik & Pfister (2024). Search-Adaptor: Embedding Customization for
    Information Retrieval. ACL 2024, 12230-12247. https://aclanthology.org/2024.acl-long.661/
[taigman] Taigman, Polyak & Wolf (2017). Unsupervised Cross-Domain Image Generation. ICLR 2017.
    arXiv:1611.02200.
[cyclegan] Zhu, Park, Isola & Efros (2017). Unpaired Image-to-Image Translation using
    Cycle-Consistent Adversarial Networks. ICCV 2017, 2223-2232. arXiv:1703.10593.
[hinton] Hinton, Vinyals & Dean (2015). Distilling the Knowledge in a Neural Network. NIPS 2014 Deep
    Learning Workshop. arXiv:1503.02531.
[rocketqav2] Ren, Qu, Liu, Zhao, She, Wu, Wang & Wen (2021). RocketQAv2: A Joint Training Method for
    Dense Passage Retrieval and Passage Re-ranking. EMNLP 2021, 2825-2835.
    https://aclanthology.org/2021.emnlp-main.224/
[selfteach] Zhuang & Zuccon (2022). CharacterBERT and Self-Teaching for Improving the Robustness of
    Dense Retrievers on Queries with Typos. SIGIR 2022, 1444-1454. doi:10.1145/3477495.3531951.
    arXiv:2204.00716.
[misspell] Sidiropoulos & Kanoulas (2022). Analysing the Robustness of Dual Encoders for Dense
    Retrieval Against Misspellings. SIGIR 2022, 2132-2136. doi:10.1145/3477495.3531818.
    arXiv:2205.02303.
[tsdae] Wang, Reimers & Gurevych (2021). TSDAE: Using Transformer-based Sequential Denoising
    Auto-Encoder for Unsupervised Sentence Embedding Learning. Findings of EMNLP 2021, 671-688.
    https://aclanthology.org/2021.findings-emnlp.59/
[dapt] Gururangan, Marasovic, Swayamdipta, Lo, Beltagy, Downey & Smith (2020). Don't Stop
    Pretraining: Adapt Language Models to Domains and Tasks. ACL 2020, 8342-8360.
    https://aclanthology.org/2020.acl-main.740/
[entityq] Sciavolino, Zhong, Lee & Chen (2021). Simple Entity-Centric Questions Challenge Dense
    Retrievers. EMNLP 2021, 6138-6148. https://aclanthology.org/2021.emnlp-main.496/
[qrewrite] Ma, Gong, He, Zhao & Duan (2023). Query Rewriting in Retrieval-Augmented Large Language
    Models. EMNLP 2023, 5303-5315. https://aclanthology.org/2023.emnlp-main.322/
[indocollex] Wibowo, Nityasya, Akyurek, Fitriany, Aji, Prasojo & Wijaya (2021). IndoCollex: A
    Testbed for Morphological Transformation of Indonesian Colloquial Words. Findings of ACL-IJCNLP
    2021, 3170-3183. https://aclanthology.org/2021.findings-acl.280/
[ndcg] Jarvelin & Kekalainen (2002). Cumulated gain-based evaluation of IR techniques. ACM TOIS
    20(4):422-446. doi:10.1145/582415.582418
[mrr] Voorhees (1999). The TREC-8 Question Answering Track Report. TREC-8, NIST SP 500-246.
[smucker] Smucker, Allan & Carterette (2007). A Comparison of Statistical Significance Tests for
    Information Retrieval Evaluation. CIKM 2007, 623-632. doi:10.1145/1321440.1321528
[north] North, Curtis & Sham (2002). A Note on the Calculation of Empirical P Values from Monte
    Carlo Procedures. Am. J. Hum. Genet. 71(2):439-441. doi:10.1086/341527
[miracl] Zhang, Thakur, Ogundepo, Kamalloo, Alfonso-Hermelo, Li, Liu, Rezagholizadeh & Lin (2023).
    MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages. TACL 11:1114-1131.
    doi:10.1162/tacl_a_00595
[bgem3] Chen, Xiao, Zhang, Luo, Lian & Liu (2024). M3-Embedding: Multi-Linguality,
    Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.
    Findings of ACL 2024, 2318-2335.
[me5] Wang, Yang, Huang, Yang, Majumder & Wei (2024). Multilingual E5 Text Embeddings: A Technical
    Report. arXiv:2402.05672.
[mpnet] Reimers & Gurevych (2020). Making Monolingual Sentence Embeddings Multilingual using
    Knowledge Distillation. EMNLP 2020. arXiv:2004.09813.
