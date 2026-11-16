"""Encoder registry and a generic frozen dense encoder (restored from the former encoder_comparison/).

Every encoder gets its own embedding cache with the same layout as methods/cache, so the existing training and
evaluation code (dataset.EmbeddingStore, dataset.TrainQueryDataset, eval.Evaluator) works unchanged.
The registry keys are the --encoder modes of the trainers and of comparison/run_all.py (bge_m3 | me5 | mpnet | arctic).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    label: str
    model_id: str
    pooling: str  # "cls" | "mean"
    query_prefix: str
    passage_prefix: str
    cache_dir: str
    query_max_length: int
    doc_max_length: int = 512
    dim: int = 1024


ENCODERS = {
    "bge_m3": EncoderSpec(
        name="bge_m3", label="BGE-M3", model_id="BAAI/bge-m3", pooling="cls",
        query_prefix="", passage_prefix="", cache_dir=os.path.join(ROOT, "methods", "cache"), query_max_length=64),
    "me5": EncoderSpec(
        name="me5", label="mE5-large-instruct", model_id="intfloat/multilingual-e5-large-instruct",
        pooling="mean",
        # Query-side instruction in the model-card format; passages are encoded without a prefix.
        query_prefix="Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery: ",
        passage_prefix="", cache_dir=os.path.join(HERE, "cache", "me5_large_instruct"),
        query_max_length=128),  # the instruction costs ~15 tokens
    # Never trained on MIRACL / Mr. TyDi (English paraphrase teacher distilled onto parallel data, 50+ languages incl.
    # Indonesian); an XLM-RoBERTa-base student with 768-d mean-pooled output, used at its released 128-token limit.
    "mpnet": EncoderSpec(
        name="mpnet", label="multilingual-mpnet", model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        pooling="mean", query_prefix="", passage_prefix="", cache_dir=os.path.join(HERE, "cache", "multilingual_mpnet"),
        query_max_length=128, doc_max_length=128, dim=768),
    # Same starting point as BGE-M3 (its RetroMAE-pretrained XLM-RoBERTa-large checkpoint), different retrieval
    # training; fine-tuned on MIRACL train in all languages (not Mr. TyDi). Model-card prefix on queries only.
    # Passages at 512 tokens like BGE-M3 / mE5, although the model takes 8192.
    "arctic": EncoderSpec(
        name="arctic", label="Arctic-Embed-L-2.0", model_id="Snowflake/snowflake-arctic-embed-l-v2.0", pooling="cls",
        query_prefix="query: ", passage_prefix="", cache_dir=os.path.join(HERE, "cache", "arctic_embed_l_v2"),
        query_max_length=64),
}


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Sentence vector from token states (not normalised): CLS token, or mean over non-padding tokens."""
    if pooling == "cls":
        return hidden[:, 0]
    mask = attention_mask.unsqueeze(-1).float()  # mean in float32: bf16 / fp16 token sums lose precision
    return (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class GenericEncoder:
    """Frozen Hugging Face encoder with configurable pooling and prefixes; L2-normalised float16 output."""

    def __init__(self, spec: EncoderSpec, device: str = "cuda", fp16: bool = True):
        self.spec = spec
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        dtype = torch.float16 if fp16 and self.device.type == "cuda" else torch.float32
        try:
            self.model = AutoModel.from_pretrained(spec.model_id, torch_dtype=dtype, attn_implementation="sdpa")
        except (ValueError, ImportError):
            self.model = AutoModel.from_pretrained(spec.model_id, torch_dtype=dtype)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        if self.model.config.hidden_size != spec.dim:
            raise ValueError(f"{spec.model_id} has hidden size {self.model.config.hidden_size}, expected {spec.dim}")

    def encode_queries(self, texts: List[str], **kwargs) -> np.ndarray:
        return self.encode([self.spec.query_prefix + t for t in texts], self.spec.query_max_length, **kwargs)

    def encode_passages(self, texts: List[str], **kwargs) -> np.ndarray:
        if self.spec.passage_prefix:
            texts = [self.spec.passage_prefix + t for t in texts]
        return self.encode(texts, self.spec.doc_max_length, **kwargs)

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int, token_budget: int = 48000, out: np.ndarray = None,
               desc: str = "encode", progress: bool = True) -> np.ndarray:
        """Length-sorted dynamic batching (batch_size * longest_sequence <= token_budget), as in model.BGEM3Encoder."""
        if out is None:
            out = np.zeros((len(texts), self.spec.dim), dtype=np.float16)
        order = np.argsort([-len(t) for t in texts], kind="stable")
        pbar = tqdm(total=len(texts), desc=desc, unit="txt", dynamic_ncols=True, disable=not progress)
        i = 0
        while i < len(order):
            longest = self.tokenizer(texts[order[i]], truncation=True, max_length=max_length)["input_ids"]
            bs = max(1, token_budget // len(longest))
            idx = order[i : i + bs]
            try:
                emb = self._encode_batch([texts[j] for j in idx], max_length)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                token_budget = max(512, token_budget // 2)
                continue
            out[idx] = emb
            i += len(idx)
            pbar.update(len(idx))
        pbar.close()
        return out

    def _encode_batch(self, batch: List[str], max_length: int) -> np.ndarray:
        enc = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).last_hidden_state.float()  # pool in float32: fp16 token sums can overflow
        return F.normalize(pool(hidden, enc["attention_mask"], self.spec.pooling), dim=-1).half().cpu().numpy()
