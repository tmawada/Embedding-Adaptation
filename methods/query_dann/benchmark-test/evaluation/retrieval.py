"""Retrieval evaluation for full-pipeline benchmarking.

Reuses existing evaluation infrastructure from the project.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, ROOT)

from methods.eval import (
    METRICS,
    dense_search,
    per_query_metrics,
    retrieval_metrics,
    retrieval_metrics_with_std,
    paired_bootstrap,
    Evaluator,
    load_head,
)
from preprocessing.dataset import CACHE_DIR, DATA_DIR, EmbeddingStore, load_qrels, load_train_pairs


@dataclass
class RetrievalResult:
    """Results from a retrieval evaluation."""
    nDCG_at_10: float
    MRR_at_10: float
    Recall_at_100: float
    per_query: Dict[str, np.ndarray]


class RetrievalEvaluator:
    """Full-pipeline retrieval evaluator using existing project infrastructure."""

    def __init__(
        self,
        cache_dir: str = os.path.join(ROOT, "methods", "cache"),
        data_dir: str = os.path.join(ROOT, "data"),
        split: str = "test",
        top_k: int = 100,
        device: Optional[torch.device] = None,
        max_queries: Optional[int] = None,
    ):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.split = split
        self.top_k = top_k
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load embedding store
        self.store = EmbeddingStore(cache_dir, device=str(self.device))
        
        # Load query splits
        with open(os.path.join(cache_dir, "splits.json")) as f:
            splits = json.load(f)
        self.qids = splits[split]
        
        # Limit queries for quick testing
        if max_queries is not None and max_queries < len(self.qids):
            self.qids = self.qids[:max_queries]

        # Initialize evaluator
        self.evaluator = Evaluator(self.store, self.qids, data_dir)

    def evaluate_baseline(self, query_type: str = "informal") -> RetrievalResult:
        """Evaluate baseline BGE-M3 (M0).

        Args:
            query_type: "formal" or "informal"
        """
        if query_type == "formal":
            embeddings = self.evaluator.formal()
        else:
            embeddings = self.evaluator.informal()

        metrics = self.evaluator.run(embeddings, k=self.top_k, with_std=False)
        per_query = self.evaluator.run_per_query(embeddings, k=self.top_k)

        return RetrievalResult(
            nDCG_at_10=metrics["nDCG@10"],
            MRR_at_10=metrics["MRR@10"],
            Recall_at_100=metrics["Recall@100"],
            per_query=per_query,
        )

    def evaluate_adapted(self, adapter_head, query_type: str = "informal") -> RetrievalResult:
        """Evaluate with adapted embeddings (M1/M2).

        Args:
            adapter_head: QueryDANN head with adapter
            query_type: "formal" or "informal"
        """
        embeddings = self.evaluator.adapted(adapter_head, which=query_type)

        metrics = self.evaluator.run(embeddings, k=self.top_k, with_std=False)
        per_query = self.evaluator.run_per_query(embeddings, k=self.top_k)

        return RetrievalResult(
            nDCG_at_10=metrics["nDCG@10"],
            MRR_at_10=metrics["MRR@10"],
            Recall_at_100=metrics["Recall@100"],
            per_query=per_query,
        )

    def get_query_texts(self) -> Dict[str, List[str]]:
        """Get query texts for the split."""
        pairs = load_train_pairs(self.data_dir).set_index("query_id")
        formal = [pairs.at[q, "formal_query"] for q in self.qids]
        informal = [pairs.at[q, "generated_informal_query"] for q in self.qids]
        return {"formal": formal, "informal": informal}

    def get_qrels(self) -> Dict[str, set]:
        """Get relevance judgments for the split."""
        return self.evaluator.qrels


def run_retrieval_benchmark(
    model,
    evaluator: RetrievalEvaluator,
    model_name: str,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, RetrievalResult]:
    """Run full retrieval benchmark for a model.

    Args:
        model: Model with encode() and adapt() methods
        evaluator: RetrievalEvaluator instance
        model_name: Name of the model condition (e.g., "M0", "M1", "M2")
        device: Device to run on
        batch_size: Batch size for encoding

    Returns:
        Dict mapping condition name to RetrievalResult
    """
    results = {}

    # Get query texts
    query_texts = evaluator.get_query_texts()
    formal_queries = query_texts["formal"]
    informal_queries = query_texts["informal"]

    def _make_result(metrics_dict, per_query_dict):
        return RetrievalResult(
            nDCG_at_10=metrics_dict["nDCG@10"],
            MRR_at_10=metrics_dict["MRR@10"],
            Recall_at_100=metrics_dict["Recall@100"],
            per_query=per_query_dict,
        )

    # M0: Baseline formal
    print(f"  Evaluating {model_name} - formal queries...")
    if hasattr(model, 'encode'):
        formal_emb = model.encode(formal_queries).cpu()
    else:
        formal_emb = evaluator.evaluator.formal()
    formal_metrics = evaluator.evaluator.run(formal_emb, k=evaluator.top_k, with_std=False)
    formal_per_query = evaluator.evaluator.run_per_query(formal_emb, k=evaluator.top_k)
    results[f"{model_name}_formal"] = _make_result(formal_metrics, formal_per_query)

    # M0/M1/M2: Baseline informal
    print(f"  Evaluating {model_name} - informal queries...")
    if hasattr(model, 'encode'):
        informal_emb = model.encode(informal_queries).cpu()
    else:
        informal_emb = evaluator.evaluator.informal()
    informal_metrics = evaluator.evaluator.run(informal_emb, k=evaluator.top_k, with_std=False)
    informal_per_query = evaluator.evaluator.run_per_query(informal_emb, k=evaluator.top_k)
    results[f"{model_name}_informal"] = _make_result(informal_metrics, informal_per_query)

    # M1/M2: Adapted informal (if model has adapter)
    if hasattr(model, 'adapt'):
        print(f"  Evaluating {model_name} - adapted informal queries...")
        # Adapter is on GPU, run adapt on GPU then move to CPU for retrieval
        adapted_emb = model.adapt(informal_emb.to(device)).cpu()
        adapted_metrics = evaluator.evaluator.run(adapted_emb, k=evaluator.top_k, with_std=False)
        adapted_per_query = evaluator.evaluator.run_per_query(adapted_emb, k=evaluator.top_k)
        results[f"{model_name}_adapted_informal"] = _make_result(adapted_metrics, adapted_per_query)

        # Robustness: Adapted formal
        print(f"  Evaluating {model_name} - adapted formal queries...")
        formal_emb_adapted = model.adapt(formal_emb.to(device)).cpu()
        adapted_formal_metrics = evaluator.evaluator.run(formal_emb_adapted, k=evaluator.top_k, with_std=False)
        adapted_formal_per_query = evaluator.evaluator.run_per_query(formal_emb_adapted, k=evaluator.top_k)
        results[f"{model_name}_adapted_formal"] = _make_result(adapted_formal_metrics, adapted_formal_per_query)

    return results


def print_retrieval_results(results: Dict[str, RetrievalResult]):
    """Print retrieval results in a formatted table."""
    print(f"\n{'Condition':<35} {'nDCG@10':>10} {'MRR@10':>10} {'Recall@100':>12}")
    print("-" * 70)
    for name, result in results.items():
        print(f"{name:<35} {result.nDCG_at_10:>10.4f} {result.MRR_at_10:>10.4f} {result.Recall_at_100:>12.4f}")