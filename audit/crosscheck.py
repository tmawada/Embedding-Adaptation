"""Credibility audit of the training pipeline: is each component taken from published research, and does the code
compute what the paper (and the spec) says?

  python3 audit/crosscheck.py            # every check, ~5 min on one GPU (loads BGE-M3 and mE5)
  python3 audit/crosscheck.py --quick    # skip the checks that load encoders (7b, 11, 12), ~1 min

Writes audit/outputs/crosscheck_report.{md,json}:
  1. Verdict          component counts per provenance verdict, and the runnable-check results
  2. Provenance       per component: source paper (verified 2026-10-19), section / equation, code location, verdict
                      (faithful / adapted / own design) and how the code differs from the paper
  3. Runnable checks  the real code run on real cached data, compared with independent reference implementations
                      written from the paper equations
  4. Original spec    where the code differs from the spec as originally given
  5. Citations        references in the original spec that are wrong or do not support what they are cited for
  6. Caveats
Nothing outside audit/outputs/ is written (check 12 re-scores into audit/outputs/repro/).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "outputs")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "methods", "query_distillation"))
sys.path.insert(0, os.path.join(ROOT, "comparison"))

from preprocessing.dataset import EmbeddingStore, TrainQueryDataset, build_triples, load_miracl_pairs, load_qrels, load_train_pairs  # noqa: E402
from methods.eval import paired_bootstrap, per_query_metrics  # noqa: E402
from methods.model import (DomainDiscriminator, LinearQueryAdapter, QueryDANN, SanitizedDomainDiscriminator,  # noqa: E402
                           ZeroInitQueryAdapter, grad_reverse, grl_alpha)
from evaluate import adapt, load_adapter  # noqa: E402  (methods/query_distillation/evaluate.py)
from evaluate_all import wrap  # noqa: E402  (comparison/evaluate_all.py: same plain-text report style)

CACHE = os.path.join(ROOT, "methods", "cache")
DATA = os.path.join(ROOT, "data")
VERIFIED = "2026-10-19"


# --------------------------------------------------------------------------- #
# Research provenance (each source opened and checked on VERIFIED; ASCII spelling of author names)
# --------------------------------------------------------------------------- #
REFS = {
    "ganin15": "Ganin & Lempitsky (2015). Unsupervised Domain Adaptation by Backpropagation. ICML 2015, PMLR 37:1180-1189. "
               "https://proceedings.mlr.press/v37/ganin15.html",
    "ganin16": "Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand & Lempitsky (2016). Domain-Adversarial "
               "Training of Neural Networks. JMLR 17(59):1-35. https://jmlr.org/papers/v17/15-239.html",
    "modir": "Xin, Xiong, Srinivasan, Sharma, Jose & Bennett (2022). Zero-Shot Dense Retrieval with Momentum Adversarial "
             "Domain Invariant Representations. Findings of ACL 2022, 4008-4020. https://aclanthology.org/2022.findings-acl.316/",
    "cpc": "van den Oord, Li & Vinyals (2018). Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748.",
    "dpr": "Karpukhin, Oguz, Min, Lewis, Wu, Edunov, Chen & Yih (2020). Dense Passage Retrieval for Open-Domain Question "
           "Answering. EMNLP 2020, 6769-6781. https://aclanthology.org/2020.emnlp-main.550/",
    "simcse": "Gao, Yao & Chen (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP 2021, 6894-6910. "
              "https://aclanthology.org/2021.emnlp-main.552/",
    "ance": "Xiong, Xiong, Li, Tang, Liu, Bennett, Ahmed & Overwijk (2021). Approximate Nearest Neighbor Negative "
            "Contrastive Learning for Dense Text Retrieval. ICLR 2021. arXiv:2007.00808.",
    "rocketqa": "Qu, Ding, Liu, Liu, Ren, Zhao, Dong, Wu & Wang (2021). RocketQA: An Optimized Training Approach to Dense "
                "Passage Retrieval for Open-Domain Question Answering. NAACL 2021, 5835-5847. "
                "https://aclanthology.org/2021.naacl-main.466/",
    "nvret": "Moreira, Osmulski, Xu, Ak, Schifferer & Oldridge (2024). NV-Retriever: Improving text embedding models with "
             "effective hard-negative mining. arXiv:2407.15831.",
    "houlsby": "Houlsby, Giurgiu, Jastrzebski, Morrone, de Laroussilhe, Gesmundo, Attariyan & Gelly (2019). "
               "Parameter-Efficient Transfer Learning for NLP. ICML 2019, PMLR 97:2790-2799.",
    "lora": "Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen (2022). LoRA: Low-Rank Adaptation of Large Language Models. "
            "ICLR 2022. arXiv:2106.09685.",
    "searchadaptor": "Yoon, Chen, Arik & Pfister (2024). Search-Adaptor: Embedding Customization for Information "
                     "Retrieval. ACL 2024, 12230-12247. https://aclanthology.org/2024.acl-long.661/",
    "taigman": "Taigman, Polyak & Wolf (2017). Unsupervised Cross-Domain Image Generation. ICLR 2017. arXiv:1611.02200.",
    "cyclegan": "Zhu, Park, Isola & Efros (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial "
                "Networks. ICCV 2017, 2223-2232. arXiv:1703.10593.",
    "hinton": "Hinton, Vinyals & Dean (2015). Distilling the Knowledge in a Neural Network. NIPS 2014 Deep Learning "
              "Workshop. arXiv:1503.02531.",
    "rocketqav2": "Ren, Qu, Liu, Zhao, She, Wu, Wang & Wen (2021). RocketQAv2: A Joint Training Method for Dense Passage "
                  "Retrieval and Passage Re-ranking. EMNLP 2021, 2825-2835. https://aclanthology.org/2021.emnlp-main.224/",
    "selfteach": "Zhuang & Zuccon (2022). CharacterBERT and Self-Teaching for Improving the Robustness of Dense Retrievers "
                 "on Queries with Typos. SIGIR 2022, 1444-1454. doi:10.1145/3477495.3531951. arXiv:2204.00716.",
    "misspell": "Sidiropoulos & Kanoulas (2022). Analysing the Robustness of Dual Encoders for Dense Retrieval Against "
                "Misspellings. SIGIR 2022, 2132-2136. doi:10.1145/3477495.3531818. arXiv:2205.02303.",
    "tasb": "Hofstaetter, Lin, Yang, Lin & Hanbury (2021). Efficiently Teaching an Effective Dense Retriever with Balanced "
            "Topic Aware Sampling. SIGIR 2021, 113-122. arXiv:2104.06967.",
    "tsdae": "Wang, Reimers & Gurevych (2021). TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for "
             "Unsupervised Sentence Embedding Learning. Findings of EMNLP 2021, 671-688. "
             "https://aclanthology.org/2021.findings-emnlp.59/",
    "dapt": "Gururangan, Marasovic, Swayamdipta, Lo, Beltagy, Downey & Smith (2020). Don't Stop Pretraining: Adapt "
            "Language Models to Domains and Tasks. ACL 2020, 8342-8360. https://aclanthology.org/2020.acl-main.740/",
    "entityq": "Sciavolino, Zhong, Lee & Chen (2021). Simple Entity-Centric Questions Challenge Dense Retrievers. "
               "EMNLP 2021, 6138-6148. https://aclanthology.org/2021.emnlp-main.496/",
    "qrewrite": "Ma, Gong, He, Zhao & Duan (2023). Query Rewriting in Retrieval-Augmented Large Language Models. "
                "EMNLP 2023, 5303-5315. https://aclanthology.org/2023.emnlp-main.322/",
    "indocollex": "Wibowo, Nityasya, Akyurek, Fitriany, Aji, Prasojo & Wijaya (2021). IndoCollex: A Testbed for "
                  "Morphological Transformation of Indonesian Colloquial Words. Findings of ACL-IJCNLP 2021, 3170-3183. "
                  "https://aclanthology.org/2021.findings-acl.280/",
    "ndcg": "Jarvelin & Kekalainen (2002). Cumulated gain-based evaluation of IR techniques. ACM TOIS 20(4):422-446. "
            "doi:10.1145/582415.582418",
    "mrr": "Voorhees (1999). The TREC-8 Question Answering Track Report. TREC-8, NIST SP 500-246.",
    "smucker": "Smucker, Allan & Carterette (2007). A Comparison of Statistical Significance Tests for Information "
               "Retrieval Evaluation. CIKM 2007, 623-632. doi:10.1145/1321440.1321528",
    "north": "North, Curtis & Sham (2002). A Note on the Calculation of Empirical P Values from Monte Carlo Procedures. "
             "Am. J. Hum. Genet. 71(2):439-441. doi:10.1086/341527",
    "miracl": "Zhang, Thakur, Ogundepo, Kamalloo, Alfonso-Hermelo, Li, Liu, Rezagholizadeh & Lin (2023). MIRACL: A "
              "Multilingual Retrieval Dataset Covering 18 Diverse Languages. TACL 11:1114-1131. doi:10.1162/tacl_a_00595",
    "bgem3": "Chen, Xiao, Zhang, Luo, Lian & Liu (2024). M3-Embedding: Multi-Linguality, Multi-Functionality, "
             "Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. Findings of ACL 2024, 2318-2335.",
    "me5": "Wang, Yang, Huang, Yang, Majumder & Wei (2024). Multilingual E5 Text Embeddings: A Technical Report. "
           "arXiv:2402.05672.",
    "mpnet": "Reimers & Gurevych (2020). Making Monolingual Sentence Embeddings Multilingual using Knowledge "
             "Distillation. EMNLP 2020. arXiv:2004.09813.",
    "arctic": "Yu, Merrick, Nuti & Campos (2024). Arctic-Embed 2.0: Multilingual Retrieval Without Compromise. "
              "arXiv:2412.04506.",
}

# (id, method, component, code location, verdict, sources, where in the source, how the code compares)
PROVENANCE = [
    ("P1", "Query-DANN", "Gradient reversal layer", "methods/model.py:87", "faithful", ["ganin15", "ganin16"],
     "ICML 2015 Sec. 3; JMLR 2016 Sec. 4.2",
     "Forward identity, backward gradient multiplied by -lambda, as in Ganin & Lempitsky 2015 (the JMLR version "
     "reverses with -1 and puts lambda in the loss; the two are equivalent for the feature extractor)."),
    ("P2", "Query-DANN", "GRL schedule alpha_p = 2/(1+exp(-10p)) - 1", "methods/model.py:108", "faithful",
     ["ganin15", "ganin16"], "JMLR 2016 Sec. 5.2.2 (gamma = 10 in all experiments)",
     "Same formula and gamma = 10, p = step / total steps (the step counter of the optimiser)."),
    ("P3", "Query-DANN", "Domain loss: binary cross-entropy, informal -> 0, formal -> 1, weight gamma = 0.1",
     "methods/query_dann/train.py:190, :193", "adapted", ["ganin16"], "JMLR 2016 Sec. 4.1 (binomial cross-entropy)",
     "Logistic / binary cross-entropy on discriminator logits, the two domains weighted 0.5 / 0.5. Difference: the "
     "loss is added with weight gamma = 0.1, so the adapter's adversarial weight is gamma x alpha_p <= 0.1 where "
     "DANN uses alpha_p. The discriminator's gradient is scaled by 0.1 as well, which AdamW largely normalises away "
     "(the discriminator gets no other gradient)."),
    ("P4", "Query-DANN", "Adversarial domain adaptation for dense retrieval", "methods/query_dann/train.py",
     "adapted", ["modir"], "Findings ACL 2022 (domain classifier + adversarial update of the DR encoder)",
     "MoDIR aligns source and target corpora by adversarially training the whole encoder (with a momentum "
     "classifier). Here the domains are informal vs formal queries and only a query adapter on a frozen encoder "
     "is trained."),
    ("P5", "Query-DANN", "Sanitized discriminator (centre + re-normalise input)", "methods/model.py:201", "own design",
     [], "-",
     "No published source found. Introduced in this project because a LayerNorm adapter output gave the "
     "discriminator a trivial 100%-accurate shortcut (docstring of ZeroInitQueryAdapter)."),
    ("P6", "Query-DANN, InfoNCE", "InfoNCE / negative log-likelihood of the positive", "methods/query_dann/train.py:187",
     "faithful", ["cpc", "dpr"], "CPC Sec. 2.3 Eq. 4; DPR Sec. 3.2 Eq. 2",
     "Cross-entropy of the positive among the candidates, cosine similarity / tau."),
    ("P7", "Query-DANN, InfoNCE", "Candidates: in-batch positives + one hard negative per query, shared by the batch",
     "preprocessing/dataset.py:141", "adapted", ["dpr", "ance"], "DPR Sec. 3.2 and 5.2; ANCE (model-mined negatives)",
     "DPR's best setting adds one BM25 negative per question and uses it as a negative for all questions in the "
     "batch. Same structure here, but the negative is mined with the dense base model (ANCE-style) instead of BM25."),
    ("P8", "Query-DANN, InfoNCE", "Temperature tau = 0.05 on cosine similarity", "methods/query_dann/train.py:184",
     "faithful", ["simcse"], "SimCSE Appendix D (tau = 0.05 with cosine similarity)", "Same value and similarity."),
    ("P9", "Query-DANN, InfoNCE", "Hard negatives from ranks 10-100, labelled positives removed",
     "methods/prepare.py:89, preprocessing/dataset.py:123", "faithful", ["nvret", "rocketqa"],
     "NV-Retriever Sec. 3.1 ('Top-K shifted by N') and 4.3.1; RocketQA Sec. 1 and 4.2.2",
     "Skipping the top-ranked candidates because they often are unlabelled positives is NV-Retriever's "
     "'shifted by N' mining (their best shifted setting discards the top 10); RocketQA reports ~70% of unlabelled "
     "top-retrieved passages are relevant."),
    ("P10", "Query-DANN, InfoNCE", "False-negative masking of other labelled positives in the denominator",
     "preprocessing/dataset.py:145", "own design", [], "-",
     "Common practice in contrastive retrieval training code; no single peer-reviewed source verified. Motivated "
     "by the false-negative problem documented in RocketQA."),
    ("P11", "Query-DANN, Distillation", "Zero-initialised residual bottleneck adapter (1024 -> 128 -> 1024)",
     "methods/model.py:152", "adapted", ["houlsby", "lora", "searchadaptor"],
     "Houlsby Sec. 2-2.1; LoRA Sec. 4.1; Search-Adaptor Sec. 4.1",
     "Bottleneck with internal skip connection and (near-)identity initialisation from Houlsby et al.; exact "
     "identity start by zero-initialising the up-projection as LoRA initialises B = 0; a residual adapter on top "
     "of frozen embeddings for retrieval as in Search-Adaptor. Differences: Search-Adaptor shares one adapter "
     "between queries and corpus, here only queries are adapted and the index is untouched."),
    ("P12", "Query-DANN, Distillation", "Linear adapter, 1024 x 1024 initialised to the identity",
     "methods/model.py:225", "own design", [], "-",
     "Ablation of the adapter architecture; not in the original spec. No source claimed."),
    ("P13", "Query-DANN, Distillation", "Formal preservation regulariser mean(1 - cos(A(z_form), z_form))",
     "methods/query_dann/train.py:191", "adapted", ["taigman", "cyclegan"],
     "Taigman et al. Sec. 4 Eq. 6 (L_TID); CycleGAN Sec. 5.2 (identity mapping loss)",
     "Both regularise the mapping to be the identity on target-domain inputs (formal queries here). They use MSE / "
     "L1; here the distance is 1 - cosine. The spec cites Su et al. (ACL 2022) for this term, which does not "
     "contain it (see Citations)."),
    ("P14", "Distillation", "Listwise KL score distillation over the teacher's top-k, with temperature",
     "methods/query_distillation/train.py:237-249", "adapted", ["rocketqav2", "selfteach", "hinton"],
     "RocketQAv2 Sec. 3.2 Eq. 2-4; Self-Teaching Sec. 4.2 Eq. 3-4; Hinton et al. (temperature)",
     "Softmax over a candidate list for teacher and student, KL between them. Differences: KL(teacher || student) "
     "(Hinton's direction; Self-Teaching uses KL(student || stop-grad teacher)); candidates are the teacher's "
     "top-100 over the full corpus instead of a training candidate list; temperature 0.05. The spec cites "
     "Hofstaetter et al. for this loss, but that paper's main loss is Margin-MSE (KL appears only in an ablation)."),
    ("P15", "Distillation", "Teacher = frozen base model on the formal query, student = adapted informal query",
     "methods/query_distillation/train.py:236", "faithful", ["selfteach", "misspell"],
     "Self-Teaching Sec. 4.2 ('the model itself is both the student and the teacher')",
     "Clean query teaches its noisy variant with the same model, as in Self-Teaching (typo -> informal). "
     "Sidiropoulos & Kanoulas pull typoed queries toward their clean version for the same goal."),
    ("P16", "TSDAE-SAPT", "TSDAE: 60% word deletion, CLS pooling, tied encoder-decoder, reconstruction CE",
     "methods/tsdae_sapt_queries/tsdae_train.py:41, :71, :143", "faithful", ["tsdae"],
     "TSDAE Sec. 3.1-3.2 and Appendix A",
     "Deletion ratio 0.6, [CLS] pooling and tied weights are the paper's recommended configuration (mE5 runs use "
     "mean pooling, the encoder's own pooling)."),
    ("P17", "TSDAE-SAPT", "Adaptive pre-training on the task's own unlabelled text", "methods/tsdae_sapt_queries/tsdae_train.py:138",
     "adapted", ["dapt"], "Gururangan et al. (TAPT)",
     "Task-adaptive pre-training on the task's unlabelled text (here: the train-split informal queries), with "
     "TSDAE instead of masked language modelling."),
    ("P18", "Query InfoNCE", "Fine-tune only the query encoder against a fixed passage index",
     "methods/query_infonce/train_infonce.py:296", "adapted", ["entityq", "dpr"],
     "Sciavolino et al. Sec. 5 ('keeping the passage encoder fixed'); DPR loss",
     "Same setting as Sciavolino et al.'s specialised question encoders, trained with the DPR / InfoNCE loss on "
     "informal queries. Not in the original spec."),
    ("P19", "LLM rewrite", "Rewrite the informal query with an LLM, then retrieve with the base encoder",
     "methods/llm_rewrite/normalize.py", "adapted", ["qrewrite", "indocollex"],
     "Ma et al. (LLM query rewriting); IndoCollex (Indonesian colloquial -> formal)",
     "Zero-shot instruction prompt (gemini-3.5-flash-lite), no trained rewriter; the task (colloquial -> formal "
     "Indonesian) is the one IndoCollex studies at the word level."),
    ("P20", "Evaluation", "nDCG@10, MRR@10, Recall@100 with binary relevance", "methods/eval.py:45", "faithful",
     ["ndcg", "mrr"], "Jarvelin & Kekalainen; TREC-8 QA (reciprocal rank)",
     "Log2 discount, ideal DCG over min(|rel|, 10) relevant passages; reciprocal rank of the first hit in the top 10."),
    ("P21", "Evaluation", "Paired bootstrap test, shift method, two-sided", "methods/eval.py:69", "faithful",
     ["smucker", "north"], "Smucker et al. Sec. 2.4; North et al. (r+1)/(n+1)",
     "Resample query pairs, shift the bootstrap distribution to mean 0, two-sided p = share with |diff| >= observed, "
     "with the (r+1)/(n+1) correction. 1,000 resamples (Smucker et al. use 100,000), so the smallest p is 0.001."),
    ("P22", "Data / encoders", "MIRACL-id, BGE-M3, mE5-large-instruct, multilingual-mpnet, Arctic-Embed-L-2.0",
     "methods/prepare.py, methods/encoders.py", "faithful", ["miracl", "bgem3", "me5", "mpnet", "arctic"], "-",
     "Public dataset and checkpoints used as released. BGE-M3, mE5 and Arctic-Embed 2.0 list MIRACL training data "
     "in their fine-tuning (see Caveats); multilingual-mpnet (2020, distilled on parallel data) predates MIRACL."),
]

# Where the code differs from the spec as originally given (archived: ../bge_dann_archive/
# informal-query-embedding-adaptation-spec.orig.md).
SPEC_DIFFS = [
    ("mE5 = multilingual-e5-large with 'query: ' / 'passage: '", "multilingual-e5-large-instruct, instruction prefix "
     "on queries, no passage prefix (methods/encoders.py)"),
    ("2,891 query pairs split 75/10/15", "3,855 pairs split 2,891 / 386 / 578, then 39 dev / test queries that "
     "repeat an earlier question dropped: 2,891 / 372 / 553 (methods/cache/splits.json, twins_excluded)"),
    ("30 epochs = ~1,020 steps", "30 x 45 steps = 1,350 steps (logs/query_dann.log)"),
    ("lr 1e-4, no warmup or clipping", "Reported runs: lr 3e-5 -> 1e-6, 10% warmup, clip 1.0 (comparison/run_all.py)"),
    ("Discriminator formula 1024 -> 512 -> 1", "1024 -> 512 -> 256 -> 1 (methods/model.py:178; the spec's own code "
     "block agrees with the code)"),
    ("Reference InfoNCE code: per-query negatives, no masking", "Shared 2B candidate pool with false-negative masking "
     "(the spec's formula text says masking; its code does not)"),
    ("LLM ~500 ms, adapter <1 ms, ~1000x", "Not in the code; measured: encoder 7.43 ms, bottleneck adapter 0.14 ms, "
     "flash-lite ~55 ms/query batched"),
    ("LLM = 'Gemini Flash', accuracy upper bound", "gemini-3.5-flash-lite; lowers formal nDCG@10 by 2.00 pts, so not "
     "an upper bound"),
    ("TSDAE on informal text (reviews or queries)", "Reported model: train-split informal queries, scored at its "
     "final step"),
    ("Methods 4-5: Stacked, UDA Query-DANN", "Code exists (methods/stacked, methods/uda_query_dann) but both were "
     "dropped from the comparison; not audited"),
    ("(not in spec)", "Query InfoNCE (full query-encoder fine-tuning) and the linear adapter are trained and reported"),
]

# References of the original spec that are wrong or do not support the claim.
CITATIONS = [
    ("Hofstaetter et al. (SIGIR 2021), 'Efficiently Teaching an Ensemble of Dense Retrievers with a Unified "
     "Knowledge Distillation Technique', 1414-1423",
     "Title and pages are wrong: 'Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware "
     "Sampling', 113-122. Its main loss is Margin-MSE (KL only in an ablation), so it is not the source of the "
     "listwise KL used here; cite RocketQAv2 / Self-Teaching instead."),
    ("Su et al. (ACL 2022), cited for the formal preservation regulariser",
     "Real paper (8352-8367), but it compares active learning, self-training and data augmentation for "
     "source-free domain adaptation; it has no identity / preservation loss. Use Taigman et al. 2017 / CycleGAN."),
    ("Reimers & Gurevych (EMNLP 2021), TSDAE, 2129-2141",
     "Authors, venue and pages are wrong: Wang, Reimers & Gurevych, Findings of EMNLP 2021, 671-688."),
    ("Karpukhin et al. (EMNLP 2020), DPR, 4171-4186",
     "Pages are wrong (4171-4186 are BERT's NAACL 2019 pages): 6769-6781."),
    ("Ryu et al. (2022), Knowledge Distillation for BERT UDA",
     "Real (arXiv:2010.11478; journal version Knowledge and Information Systems, doi:10.1007/s10115-022-01736-y), "
     "but it combines adversarial adaptation with classifier distillation for sentiment; background only, not the "
     "source of the score-distillation loss."),
    ("Ganin et al. (JMLR 2016); Gururangan et al. (ACL 2020); Pfeiffer et al. (EMNLP 2020 demos); Koto et al. "
     "(EMNLP 2021)", "Correct as cited."),
    ("Sandriago et al. (ICDEES 2026)", "Not verifiable online (the author's own paper); not checked."),
]

CAVEATS = [
    "Contamination: all queries come from the official MIRACL-id train split, and BGE-M3, mE5 and Arctic-Embed 2.0 "
    "list MIRACL training data in their fine-tuning, so their formal scores are probably inflated (multilingual-mpnet "
    "predates MIRACL). Method comparisons share the exposure; absolute numbers are not comparable to published MIRACL "
    "results.",
    "Duplicate questions (resolved 2026-10-19): MIRACL-id train repeats some questions under different ids. The 39 "
    "dev / test queries that repeated an earlier question (14 dev, 25 test) were dropped (methods/drop_twins.py; ids in "
    "splits.json twins_excluded; the old split is kept as splits_with_twins.json) and every model was retrained. "
    "Before that, re-scoring without the 24 test twins had dropped Query-DANN (bottleneck) on BGE-M3 from +1.76 to "
    "+1.27 pts (no longer significant) while the distillation gains barely moved.",
    "Single seed (42) for the reported numbers. The five-seed run was stopped before it finished, and its seed "
    "runs were selected on the dev split that still contained repeated questions: a later --std true run needs "
    "--force.",
    "tau = 0.05 was chosen for BGE-M3; mE5's compressed cosine scores may put it at a disadvantage.",
    "Checkpoints are selected on dev informal nDCG@10 (dev labels); TSDAE is reported at its final step, a choice "
    "made after seeing that dev selection keeps step 50.",
    "Training runs adapter / discriminator forward passes under bf16 autocast; check 4 runs the same code in fp32 so "
    "the reference comparison is exact. The distillation teacher scores the corpus in fp16 (check 6 measures the "
    "effect on its top-100).",
    "normalize.py records 'shots': 3 for every cached rewrite, but the batched requests used for the reported "
    "rewrites carry no few-shot examples (do_batch builds the prompt from INSTRUCTION only): the LLM row is zero-shot.",
]


# --------------------------------------------------------------------------- #
# Runnable checks
# --------------------------------------------------------------------------- #
CHECKS = []


def check(cid: str, name: str, code: str, slow: bool = False):
    def register(fn):
        CHECKS.append({"id": cid, "name": name, "code": code, "slow": slow, "fn": fn})
        return fn
    return register


def load_module(name: str, path: str):
    """Import a script by path (two trainers are both called train.py)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source(path: str) -> str:
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


class Context:
    """Shared data, loaded on first use."""

    def __init__(self):
        self._store = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(os.path.join(CACHE, "splits.json")) as f:
            self.splits = json.load(f)
        with open(os.path.join(CACHE, "hard_negs.json")) as f:
            self.hard_negs = json.load(f)
        self.triples = build_triples(DATA, self.hard_negs)

    @property
    def store(self) -> EmbeddingStore:
        if self._store is None:
            self._store = EmbeddingStore(CACHE, device=self.device)
        return self._store

    def train_dataset(self) -> TrainQueryDataset:
        return TrainQueryDataset(self.triples, self.splits["train"], self.store, hn_range=(10, 100), seed=42)


@check("1", "Adapters start as the identity; parameter counts", "methods/model.py:152, :225")
def c_adapters(ctx):
    torch.manual_seed(0)
    z = F.normalize(torch.randn(512, 1024), dim=-1)
    errs, counts = [], {}
    for name, cls in (("bottleneck", ZeroInitQueryAdapter), ("linear", LinearQueryAdapter)):
        a = cls()
        for training in (True, False):  # dropout on and off
            a.train(training)
            with torch.no_grad():
                errs.append((a(z) - z).abs().max().item())
        counts[name] = sum(p.numel() for p in a.parameters())
    ok = max(errs) < 1e-6 and counts == {"bottleneck": 263552, "linear": 1049600}
    return ok, (f"max |A(z) - z| = {max(errs):.1e} in train and eval mode; parameters bottleneck "
                f"{counts['bottleneck']:,}, linear {counts['linear']:,}"), max(errs)


@check("2", "Gradient reversal: identity forward, -alpha x gradient backward; alpha_p schedule",
       "methods/model.py:87, :108")
def c_grl(ctx):
    torch.manual_seed(0)
    x = torch.randn(64, 1024, requires_grad=True)
    g = torch.randn(64, 1024)
    fwd, bwd = 0.0, 0.0
    for alpha in (0.0, 0.3, 1.0):
        x.grad = None
        y = grad_reverse(x, alpha)
        fwd = max(fwd, (y - x).abs().max().item())
        (y * g).sum().backward()
        bwd = max(bwd, (x.grad + alpha * g).abs().max().item())
    ps = np.linspace(0, 1, 21)
    sched = max(abs(grl_alpha(p) - (2 / (1 + math.exp(-10 * p)) - 1)) for p in ps)
    err = max(fwd, bwd, sched)
    return err < 1e-6, (f"forward error {fwd:.1e}, backward error {bwd:.1e} (alpha 0 / 0.3 / 1), schedule error "
                        f"{sched:.1e}; alpha(0) = {grl_alpha(0):.3f}, alpha(1) = {grl_alpha(1):.5f}"), err


@check("3", "Sanitized discriminator ignores a constant shift and the scale of its input", "methods/model.py:201")
def c_disc(ctx):
    torch.manual_seed(0)
    z = F.normalize(torch.randn(256, 1024), dim=-1)
    san, plain = SanitizedDomainDiscriminator().eval(), DomainDiscriminator().eval()
    plain.load_state_dict(san.state_dict())  # same weights, without the centring
    with torch.no_grad():
        base = san(z)
        shifts = [z + c for c in (0.05, -0.2, 1.0)] + [3.0 * z]
        err = max((san(s) - base).abs().max().item() for s in shifts)
        plain_err = max((plain(s) - plain(z)).abs().max().item() for s in shifts)
    return err < 1e-5, (f"max logit change {err:.1e} under z + c and 3z (the same network without centring changes "
                        f"by {plain_err:.2f}, so the centring is what removes the shortcut)"), err


@check("4", "Query-DANN train_step losses equal independent per-query reference losses", "methods/query_dann/train.py:172")
def c_dann_losses(ctx):
    from accelerate import Accelerator
    qd = load_module("audit_query_dann_train", "methods/query_dann/train.py")
    ds, store = ctx.train_dataset(), ctx.store
    head = QueryDANN().to(store.device)
    ckpt = os.path.join(ROOT, "methods", "query_dann", "outputs", "harmonized", "best")
    head.adapter.load_state_dict(torch.load(os.path.join(ckpt, "adapter.pt"), map_location=store.device))
    head.discriminator.load_state_dict(torch.load(os.path.join(ckpt, "discriminator.pt"), map_location=store.device))
    head.eval()  # dropout off, so the code and the reference see the same network
    items = [ds[i] for i in random.Random(0).sample(range(len(ds)), 64)]
    batch, B = ds.collate(items), len(items)
    args = Namespace(tau=0.05, gamma=0.1, mu=0.5, eta=10.0, domain_targets="formal", grad_clip=0.0)
    opt = torch.optim.SGD(head.parameters(), lr=0.0)  # lr 0: the call leaves the weights unchanged

    class Silent:
        def log(self, *a, **k):
            pass

    loss, nce, dom, pres, alpha, _, _ = qd.train_step(batch, head, opt, None, Accelerator(mixed_precision="no"),
                                                     Silent(), args, 500, 1350, store.device)
    with torch.no_grad():  # reference, written from the equations with its own masking rule
        zq = F.normalize(head.adapter(batch["z_inf"]), dim=-1).double()
        zp = F.normalize(head.adapter(batch["z_form"]), dim=-1).double()
        cand, rows = batch["z_cand"].double(), [it["pos_row"] for it in items] + [it["neg_row"] for it in items]
        terms = []
        for i, it in enumerate(items):
            keep = [j for j in range(2 * B) if j == i or rows[j] not in set(it["pos_set"])]
            logits = torch.stack([zq[i] @ cand[j] for j in keep]) / args.tau
            terms.append(torch.logsumexp(logits, 0) - logits[keep.index(i)])
        ref_nce = torch.stack(terms).mean().item()
        li = head.discriminator(zq.float()).double()  # GRL is the identity in the forward pass
        lf = head.discriminator(batch["z_form"]).double()
        ref_dom = (0.5 * torch.log1p(torch.exp(li)).mean() + 0.5 * torch.log1p(torch.exp(-lf)).mean()).item()
        ref_pres = (1 - (zp * batch["z_form"].double()).sum(-1)).mean().item()
    ref_total = ref_nce + args.gamma * ref_dom + args.mu * ref_pres
    errs = {"InfoNCE": abs(nce.item() - ref_nce), "domain": abs(dom.item() - ref_dom),
            "preserve": abs(pres.item() - ref_pres), "total": abs(loss.item() - ref_total)}
    err = max(errs.values())
    return err < 1e-4, (f"code vs reference: InfoNCE {nce.item():.5f} / {ref_nce:.5f}, domain {dom.item():.5f} / "
                        f"{ref_dom:.5f}, preserve {pres.item():.6f} / {ref_pres:.6f}; max abs diff {err:.1e} "
                        f"(trained seed-42 checkpoint, 64 train queries, fp32)"), err


@check("5", "Batch construction: negatives from ranks 10-100, never positives; masking rule",
       "preprocessing/dataset.py:111-160, methods/prepare.py:89")
def c_batches(ctx):
    ds, store = ctx.train_dataset(), ctx.store
    qrels = load_qrels(DATA)
    pos_in_negs = sum(len(set(ctx.hard_negs.get(q, [])) & qrels.get(q, set())) for q in ctx.hard_negs)
    bad_neg, bad_pos, n_items = 0, 0, 0
    for i, qid in enumerate(ds.qids):  # one full epoch of sampling
        it = ds[i]
        allowed = {store.doc2row[d] for d in ctx.hard_negs[qid][10:100] if d in store.doc2row}
        bad_neg += it["neg_row"] not in allowed
        bad_pos += it["neg_row"] in set(it["pos_set"])
        n_items += 1
    rng, mask_err, masked = random.Random(1), 0, 0
    for _ in range(30):
        items = [ds[i] for i in rng.sample(range(len(ds)), 64)]
        got = ds.collate(items)["false_neg"].cpu()
        rows = [it["pos_row"] for it in items] + [it["neg_row"] for it in items]
        want = torch.tensor([[j != i and rows[j] in set(it["pos_set"]) for j in range(len(rows))]
                             for i, it in enumerate(items)])
        mask_err += int((got != want).sum())
        masked += int(want.sum())
        mask_err += int(got.diagonal().sum())  # a query's own target must never be masked
    ok = pos_in_negs == 0 and bad_neg == 0 and bad_pos == 0 and mask_err == 0
    return ok, (f"labelled positives in mined negative lists: {pos_in_negs}; over one epoch ({n_items} queries) "
                f"negatives outside ranks 10-100: {bad_neg}, negatives that are positives: {bad_pos}; masking rule "
                f"mismatches over 30 batches: {mask_err} ({masked} entries legitimately masked)"), mask_err + bad_neg


@check("6", "Distillation: KL expression equals sum p_t log(p_t / p_s) / B; teacher candidates are its top-100",
       "methods/query_distillation/train.py:237-249")
def c_kd(ctx):
    src = source("methods/query_distillation/train.py")
    expressions = ["(z_form.half() @ store.corpus.T).float().topk(args.kd_topk, dim=1)",
                   "F.kl_div(F.log_softmax(s_scores / args.kd_tau, dim=-1),",
                   "F.softmax(t_scores / args.kd_tau, dim=-1), reduction=\"batchmean\")",
                   "loss = loss_kd + args.w_infonce * loss_nce + args.mu * loss_pres"]
    missing = [e for e in expressions if e not in src]
    ds, store = ctx.train_dataset(), ctx.store
    items = [ds[i] for i in random.Random(2).sample(range(len(ds)), 64)]
    batch, tau = ds.collate(items), 0.05
    adapter = load_adapter(os.path.join(ROOT, "methods", "query_distillation", "outputs", "harmonized", "best"),
                           store.device)
    with torch.no_grad():
        z_form = batch["z_form"]
        t_scores, t_idx = (z_form.half() @ store.corpus.T).float().topk(100, dim=1)  # the code's teacher
        cand = store.corpus[t_idx].float()
        s_scores = torch.einsum("bd,bkd->bk", adapt(adapter, batch["z_inf"]), cand)
        code = F.kl_div(F.log_softmax(s_scores / tau, dim=-1), F.softmax(t_scores / tau, dim=-1),
                        reduction="batchmean").item()
        p_t, p_s = F.softmax(t_scores.double() / tau, -1), F.softmax(s_scores.double() / tau, -1)
        ref = (p_t * (p_t.log() - p_s.log())).sum(-1).mean().item()
        corpus32 = store.corpus.float()
        exact = torch.cat([(z_form[i:i + 16] @ corpus32.T).topk(100, dim=1).indices for i in range(0, len(z_form), 16)])
        del corpus32
        overlap = np.mean([len(set(a.tolist()) & set(b.tolist())) / 100 for a, b in zip(t_idx, exact)])
    err = abs(code - ref)
    ok = not missing and err < 1e-6 and overlap >= 0.99
    return ok, (f"the 4 loss expressions are present in the code: {not missing}; KL code {code:.6f} vs manual "
                f"{ref:.6f} (diff {err:.1e}); teacher top-100 in fp16 vs exact fp32 overlap {100 * overlap:.2f}%"), err


@check("7a", "TSDAE noise deletes ~60% of words", "methods/tsdae_sapt_queries/tsdae_train.py:41")
def c_tsdae_noise(ctx):
    tsdae = load_module("audit_tsdae_train", "methods/tsdae_sapt_queries/tsdae_train.py")
    tp = load_train_pairs(DATA).set_index("query_id")
    rng = np.random.default_rng(42)
    total = kept = 0
    for q in ctx.splits["train"]:
        text = tp.at[q, "generated_informal_query"]
        total += len(text.split())
        kept += len(tsdae.delete_words(text, 0.6, rng).split())
    frac = 1 - kept / total
    return 0.55 <= frac <= 0.61, (f"{100 * frac:.1f}% of {total:,} words deleted over the {len(ctx.splits['train'])} "
                                  f"train informal queries (slightly below 60% because one word is always kept)"), frac


@check("7b", "TSDAE decoder is tied to the encoder; CLS pooling", "methods/tsdae_sapt_queries/tsdae_train.py:71, :143",
       slow=True)
def c_tsdae_tied(ctx):
    from sentence_transformers.losses import DenoisingAutoEncoderLoss
    tsdae = load_module("audit_tsdae_train", "methods/tsdae_sapt_queries/tsdae_train.py")
    model = tsdae.build_model("BAAI/bge-m3", 128, "cls")
    loss = DenoisingAutoEncoderLoss(model, decoder_name_or_path="BAAI/bge-m3", tie_encoder_decoder=True)
    enc = {id(p) for p in model[0].auto_model.parameters()}
    dec = list(loss.decoder.parameters())
    tied = sum(id(p) in enc for p in dec)
    cls = bool(model[1].pooling_mode_cls_token)
    del loss, model
    return tied > 0 and cls, (f"{tied} of {len(dec)} decoder parameter tensors are the encoder's own tensors (the rest "
                              f"are the new cross-attention and LM head); pooling = CLS: {cls}"), float(tied)


@check("8", "Metrics equal an independent implementation, scikit-learn's ndcg_score and hand-computed cases",
       "methods/eval.py:45")
def c_metrics(ctx):
    from sklearn.metrics import ndcg_score
    hand = [  # (relevant, ranking, nDCG@10, MRR@10, Recall@100)
        ({"a"}, ["x", "a"], 1 / math.log2(3), 0.5, 1.0),
        ({"a", "b"}, ["a", "x", "b"], (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)), 1.0, 1.0),
        ({"a"}, [f"x{i}" for i in range(10)] + ["a"], 0.0, 0.0, 1.0),
        ({"a", "b", "c"}, [f"x{i}" for i in range(49)] + ["a"] + [f"y{i}" for i in range(50)], 0.0, 0.0, 1 / 3),
    ]
    got = per_query_metrics({str(i): r for i, (_, r, *_x) in enumerate(hand)}, {str(i): rel for i, (rel, *_x) in enumerate(hand)})
    hand_err = max(abs(got[m][i] - h[2 + k]) for i, h in enumerate(hand)
                   for k, m in enumerate(("nDCG@10", "MRR@10", "Recall@100")))
    rng = np.random.default_rng(0)
    ranked, qrels = {}, {}
    for q in range(500):
        rel = set(rng.choice(1000, size=rng.integers(1, 6), replace=False).tolist())
        ranked[str(q)] = [int(d) for d in rng.permutation(1000)[:100]]
        qrels[str(q)] = rel
    code = per_query_metrics(ranked, qrels)
    ind = {"nDCG@10": [], "MRR@10": [], "Recall@100": []}
    sk = []
    for q, docs in ranked.items():
        rel = qrels[q]
        gains = np.array([d in rel for d in docs], dtype=float)
        disc = 1 / np.log2(np.arange(2, 102))
        ind["nDCG@10"].append((gains[:10] * disc[:10]).sum() / disc[:min(len(rel), 10)].sum())
        hits = np.flatnonzero(gains[:10])
        ind["MRR@10"].append(1 / (hits[0] + 1) if len(hits) else 0.0)
        ind["Recall@100"].append(gains.sum() / len(rel))
        missing = [d for d in rel if d not in docs]  # unretrieved relevant passages, scored below every ranked one
        y_true = np.r_[gains, np.ones(len(missing))][None]
        y_score = np.r_[-np.arange(100, dtype=float), -1e9 - np.arange(len(missing))][None]
        sk.append(ndcg_score(y_true, y_score, k=10))
    rand_err = max(np.abs(code[m] - np.array(v)).max() for m, v in ind.items())
    sk_err = np.abs(code["nDCG@10"] - np.array(sk)).max()
    err = max(hand_err, rand_err, sk_err)
    return err < 1e-9, (f"hand cases max error {hand_err:.1e}; 500 random queries vs independent code {rand_err:.1e}, "
                        f"vs sklearn ndcg_score {sk_err:.1e}"), float(err)


@check("9", "Paired bootstrap: shift method, p = 1 for identical systems, ~5% false positives under the null",
       "methods/eval.py:69")
def c_bootstrap(ctx):
    rng = np.random.default_rng(0)
    a = rng.random(578)
    same = paired_bootstrap(a, a.copy())
    b = a - 0.05 + 0.02 * rng.standard_normal(578)
    shifted = paired_bootstrap(a, b)
    # Independent shift-method implementation (Smucker et al. 2007, Sec. 2.4) with the same resamples.
    d = a - b
    idx = np.random.default_rng(0).integers(0, len(d), size=(1000, len(d)))
    boot = d[idx].mean(1)
    null = boot - d.mean()  # shift the bootstrap distribution to mean 0
    ref_p = (np.sum(np.abs(null) >= abs(d.mean())) + 1) / 1001
    ref_ci = np.percentile(boot, [2.5, 97.5])
    ref_err = max(abs(shifted["p"] - ref_p), abs(shifted["ci95_low"] - ref_ci[0]), abs(shifted["ci95_high"] - ref_ci[1]))
    false_pos = np.mean([paired_bootstrap(rng.random(578), rng.random(578), seed=s)["p"] < 0.05 for s in range(300)])
    ok = same["p"] == 1.0 and shifted["p"] <= 0.001 and ref_err < 1e-12 and 0.02 <= false_pos <= 0.09
    return ok, (f"identical systems p = {same['p']:.3f}; clear shift p = {shifted['p']:.4f}; vs independent shift-"
                f"method code diff {ref_err:.1e}; false-positive rate at p < 0.05 over 300 null pairs "
                f"{100 * false_pos:.1f}%"), float(ref_err)


@check("10", "No leakage: disjoint splits, no repeated questions in dev / test, trainers read only train",
       "methods/cache/splits.json, preprocessing/dataset.py drop_twins, the four trainers, normalize.py")
def c_leakage(ctx):
    s = ctx.splits
    tr, dv, te = set(s["train"]), set(s["dev"]), set(s["test"])
    with open(os.path.join(CACHE, "corpus_ids.json")) as f:
        in_corpus = set(json.load(f))
    answerable = {p["query_id"] for p in load_miracl_pairs(DATA) if p["positive_doc_id"] in in_corpus}
    excluded = set(s.get("twins_excluded", []))
    overlap = len(tr & dv) + len(tr & te) + len(dv & te) + len((tr | dv | te) & excluded)
    covered = (tr | dv | te | excluded) == answerable
    ds_outside = len(set(ctx.train_dataset().qids) - tr)
    trainers = ["methods/query_dann/train.py", "methods/query_distillation/train.py",
                "methods/tsdae_sapt_queries/tsdae_train.py", "methods/query_infonce/train_infonce.py"]
    test_refs = {t: sum(source(t).count(k) for k in ('["test"]', "['test']")) for t in trainers}
    tsdae_train_only = 'sentences = [tp.at[q, "generated_informal_query"] for q in splits["train"]]' in source(trainers[2])
    norm = source("methods/llm_rewrite/normalize.py")
    shots_train_only = 'random.Random(args.seed).sample(splits["train"], args.shots)' in norm
    # Repeated questions, counted independently of drop_twins: same formal or informal text as a train query, or as
    # another dev / test query.
    tp = load_train_pairs(DATA).set_index("query_id")
    cols = ("formal_query", "generated_informal_query")
    key = lambda q, col: " ".join(str(tp.at[q, col]).lower().split())
    train_keys = {c: {key(q, c) for q in tr} for c in cols}
    twins = {name: sum(any(key(q, c) in train_keys[c] for c in cols) for q in s[name]) for name in ("dev", "test")}
    held_out = sorted(dv | te, key=int)
    repeats = sum(len({key(q, c) for q in held_out}) < len(held_out) for c in cols)
    ok = (overlap == 0 and covered and ds_outside == 0 and not any(test_refs.values()) and tsdae_train_only
          and shots_train_only and not any(twins.values()) and repeats == 0)
    return ok, (f"split overlaps {overlap}; train + dev + test + {len(excluded)} excluded twins = the "
                f"{len(answerable):,} answerable queries: {covered}; sizes {len(tr)} / {len(dv)} / {len(te)}; training "
                f"queries outside train: {ds_outside}; references to the test split in the trainers: "
                f"{sum(test_refs.values())}; TSDAE text = train informal queries: {tsdae_train_only}; LLM few-shot pool "
                f"= train: {shots_train_only}; dev / test queries repeating a train question: {twins['dev']} / "
                f"{twins['test']}; repeats among dev + test: {repeats}"), float(overlap + sum(twins.values()) + repeats)


@check("11", "Cached query and passage vectors are reproduced by every encoder (BGE-M3, Arctic CLS; mE5, mpnet mean)",
       "methods/model.py:20, methods/encoders.py", slow=True)
def c_encoders(ctx):
    from methods.encoders import ENCODERS, GenericEncoder
    from methods.model import BGEM3Encoder
    from preprocessing.dataset import load_corpus
    tp = load_train_pairs(DATA).set_index("query_id")
    qids = random.Random(3).sample(ctx.splits["test"], 100)
    doc_ids, passages = load_corpus(DATA)
    pick = random.Random(4).sample(range(len(doc_ids)), 50)
    texts = {"formal": [tp.at[q, "formal_query"] for q in qids], "informal": [tp.at[q, "generated_informal_query"] for q in qids]}
    lines, worst = [], 1.0
    for enc_name in ENCODERS:
        spec = ENCODERS[enc_name]
        store = ctx.store if enc_name == "bge_m3" else EmbeddingStore(spec.cache_dir, device=ctx.device)
        rows = [store.qid2row[q] for q in qids]
        if enc_name == "bge_m3":
            enc = BGEM3Encoder(spec.model_id)
            q_emb = lambda t: enc.encode(t, max_length=spec.query_max_length, progress=False)
            p_emb = lambda t: enc.encode(t, max_length=512, progress=False)
        else:
            enc = GenericEncoder(spec)
            q_emb = lambda t: enc.encode_queries(t, progress=False)
            p_emb = lambda t: enc.encode_passages(t, progress=False)
        cos = []
        for side in ("formal", "informal"):
            new = torch.from_numpy(q_emb(texts[side])).float().to(store.device)
            cached = getattr(store, side)[rows].float()
            cos.append((new * cached).sum(-1).min().item())
        prow = [store.doc2row[doc_ids[i]] for i in pick]
        newp = torch.from_numpy(p_emb([passages[i] for i in pick])).float().to(store.device)
        cos.append((newp * store.corpus[prow].float()).sum(-1).min().item())
        worst = min(worst, *cos)
        lines.append(f"{spec.label}: min cosine formal {cos[0]:.5f}, informal {cos[1]:.5f}, passages {cos[2]:.5f}")
        del enc
        torch.cuda.empty_cache()
    return worst >= 0.995, "; ".join(lines) + " (100 test queries, 50 passages)", worst


@check("12", "Reported numbers are reproduced from the saved checkpoints", "comparison/evaluate_all.py", slow=True)
def c_repro(ctx):
    import evaluate_all
    from methods.encoders import ENCODERS
    from run_all import report_name
    out_dir = os.path.join(OUT, "repro")
    lines, ok = [], True
    for enc, report in ((e, report_name(e)) for e in ENCODERS):
        payload = evaluate_all.evaluate(enc, Namespace(split="test", bootstrap=1000, out_dir=out_dir))
        with open(os.path.join(ROOT, "comparison", "outputs", report + ".json")) as f:
            saved = json.load(f)["results"]
        same = json.loads(json.dumps(payload["results"])) == saved
        ok &= same
        lines.append(f"{report}.json identical: {same}")
    return ok, "; ".join(lines), None


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def run_checks(quick: bool) -> list:
    ctx, results = Context(), []
    for c in CHECKS:
        if quick and c["slow"]:
            results.append({**{k: c[k] for k in ("id", "name", "code")}, "status": "SKIPPED", "detail": "--quick",
                            "value": None, "seconds": 0.0})
            continue
        t0 = time.time()
        print(f"[check {c['id']}] {c['name']}", flush=True)
        try:
            ok, detail, value = c["fn"](ctx)
            status = "INFO" if ok is None else "PASS" if ok else "FAIL"  # INFO: a measurement, not a pass/fail test
        except Exception as e:  # a crash is a failed check, with the reason in the report
            status, detail, value = "FAIL", f"crashed: {type(e).__name__}: {e}", None
            traceback.print_exc()
        print(f"    {status}: {detail}", flush=True)
        results.append({"id": c["id"], "name": c["name"], "code": c["code"], "status": status, "detail": detail,
                        "value": value, "seconds": round(time.time() - t0, 1)})
    return results


def table(headers, rows, numeric=()) -> list:
    """Padded markdown table that also lines up in a plain-text editor; text left-aligned, `numeric` columns right."""
    widths = [max(len(r[i]) for r in [headers, *rows]) for i in range(len(headers))]
    line = lambda r: "| " + " | ".join(c.rjust(w) if i in numeric else c.ljust(w)
                                      for i, (c, w) in enumerate(zip(r, widths))) + " |"
    rule = "|" + "|".join("-" * (w + 1) + ":" if i in numeric else "-" * (w + 2) for i, w in enumerate(widths)) + "|"
    return [line(headers), rule, *map(line, rows)]


def markdown(results: list) -> str:
    verdicts = {}
    for p in PROVENANCE:
        verdicts.setdefault(p[4], []).append(p[0])
    n_pass = sum(r["status"] == "PASS" for r in results)
    n_run = sum(r["status"] in ("PASS", "FAIL") for r in results)
    n_skip = sum(r["status"] == "SKIPPED" for r in results)
    n_info = sum(r["status"] == "INFO" for r in results)
    cited = sorted({s for p in PROVENANCE for s in p[5]}, key=list(REFS).index)
    lines = ["# Credibility audit: training pipeline vs published research and the spec", "",
             f"- Scope:     Query-DANN, query distillation, TSDAE-SAPT, Query InfoNCE, LLM rewrite, data, evaluation",
             f"- Sources:   every reference below was opened and checked on {VERIFIED}",
             "- Code:      the checks import and run the project's own code on the real cached data",
             "- Rerun:     python3 audit/crosscheck.py", "",
             "## 1. Verdict", "",
             *table(["Provenance verdict", "Components", "Ids"],
                    [[v, str(len(ids)), ", ".join(ids)] for v, ids in
                     sorted(verdicts.items(), key=lambda kv: ["faithful", "adapted", "own design"].index(kv[0]))],
                    numeric=(1,)),
             "",
             wrap(f"Runnable checks: {n_pass} of {n_run} pass" + (f" ({n_skip} skipped with --quick)" if n_skip else "")
                  + (f"; {n_info} INFO measurement(s), see section 3" if n_info else "") + "."),
             "",
             wrap("faithful = the component is the published method with the published settings; adapted = taken from "
                  "the cited work with the stated changes; own design = no published source was found, the component "
                  "is this project's choice (said so explicitly rather than given a citation it does not have)."),
             "", "## 2. Research provenance", "",
             *table(["Id", "Component", "Verdict", "Sources"],
                    [[p[0], p[2] if len(p[2]) <= 52 else p[2][:49] + "...", p[4], ", ".join(p[5]) or "-"]
                     for p in PROVENANCE]), ""]
    for pid, method, comp, code, verdict, srcs, where, how in PROVENANCE:
        lines += [wrap(f"{pid}. {comp} ({method}; {code}) - {verdict}.", "    "),
                  wrap(f"    Source: {where}." if srcs else "    Source: none found.", "    "),
                  wrap(f"    {how}", "    "), ""]
    lines += ["## 3. Runnable checks", "",
              *table(["Check", "Result", "Name"], [[r["id"], r["status"], r["name"] if len(r["name"]) <= 72 else r["name"][:69] + "..."]
                                                         for r in results]), ""]
    for r in results:
        lines += [wrap(f"{r['id']}. {r['name']} [{r['code']}]: {r['status']}.", "    "),
                  wrap(f"    {r['detail']}", "    "), ""]
    lines += ["## 4. Code vs the original spec", "",
              wrap("Spec as originally given (archived as ../bge_dann_archive/informal-query-embedding-adaptation-"
                   "spec.orig.md). Everything not listed matches: adapter, GRL and its schedule, gamma 0.1, mu 0.5, "
                   "eta 10, tau 0.05, hard negatives 10-100 with masking (formula text), score KD (k 100, tau_kd "
                   "0.05), TSDAE deletion 0.6, metrics and bootstrap."), ""]
    for a, b in SPEC_DIFFS:
        lines += [wrap(f"- Spec: {a}", "  "), wrap(f"  Code: {b}", "  ")]
    lines += ["", "## 5. Citations in the original spec", ""]
    for a, b in CITATIONS:
        lines += [wrap(f"- {a}", "  "), wrap(f"  -> {b}", "  ")]
    lines += ["", "## 6. Caveats", "", *[wrap(f"- {c}", "  ") for c in CAVEATS],
              "", "## References (verified)", "", *[wrap(f"[{k}] {REFS[k]}", "    ") for k in cited]]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Skip the checks that load full encoders (7b, 11, 12)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    results = run_checks(args.quick)
    with open(os.path.join(OUT, "crosscheck_report.json"), "w") as f:
        json.dump({"verified": VERIFIED, "provenance": [dict(zip(("id", "method", "component", "code", "verdict",
                                                                  "sources", "where", "how"), p)) for p in PROVENANCE],
                   "references": REFS, "checks": results, "spec_differences": SPEC_DIFFS, "citations": CITATIONS,
                   "caveats": CAVEATS}, f, indent=2)
    with open(os.path.join(OUT, "crosscheck_report.md"), "w") as f:
        f.write(markdown(results))
    graded = [r for r in results if r["status"] in ("PASS", "FAIL")]
    print(f"-> {os.path.relpath(os.path.join(OUT, 'crosscheck_report.md'), ROOT)} "
          f"({sum(r['status'] == 'PASS' for r in graded)} / {len(graded)} checks pass)")


if __name__ == "__main__":
    main()
