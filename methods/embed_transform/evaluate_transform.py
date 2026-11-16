"""Evaluate transformed embeddings for informal query retrieval.

This script:
1. Loads a trained transform from disk
2. Extracts embeddings for informal queries using frozen BGE-M3
3. Applies the transform to map to formal-equivalent space
4. Evaluates retrieval performance using the existing eval framework
5. Compares against baselines (base BGE-M3 formal, base BGE-M3 informal, LLM-Norm)

Usage:
    python3 models/embed_transform/evaluate_transform.py       # Use default linear transform
    python3 models/embed_transform/evaluate_transform.py --transform_type whitening_procrustes
    python3 models/embed_transform/evaluate_transform.py --transform_type mlp
    python3 methods/embed_transform/evaluate_transform.py --transform_dir methods/embed_transform/outputs/transforms/linear
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# Add root to path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from preprocessing.dataset import EmbeddingStore, load_train_pairs  # noqa: E402
from methods.eval import METRICS, Evaluator  # noqa: E402
from methods.model import BGEM3Encoder  # noqa: E402


def load_transform(transform_dir, transform_type, root):
    """Load a trained transform from disk."""
    transform_path = os.path.join(transform_dir, "transform_config.json")

    with open(transform_path) as f:
        config = json.load(f)

    transform_type = config["transform_type"]
    input_dim = config["input_dim"]

    if transform_type == "linear":
        params = config["transform_params"]
        W = np.array(params["W"], dtype=np.float32)
        b = np.array(params["b"], dtype=np.float32)
        transform = LinearTransform(input_dim=input_dim)
        transform.W = W
        transform.b = b
        return transform, "linear"

    elif transform_type == "whitening_procrustes":
        params = config["transform_params"]
        W = np.array(params["W"], dtype=np.float32)
        mean_inf = np.array(params["mean_inf"], dtype=np.float32) if params.get("mean_inf") is not None else None
        mean_form = np.array(params["mean_form"], dtype=np.float32) if params.get("mean_form") is not None else None
        whitening_matrix = np.array(params["whitening_matrix"], dtype=np.float32) if params.get("whitening_matrix") is not None else None

        transform = WhiteningProcrustesTransform()
        transform.W = W
        transform.mean_inf = mean_inf
        transform.mean_form = mean_form
        transform.whitening_matrix = whitening_matrix
        return transform, "whitening_procrustes"

    elif transform_type == "mlp":
        import torch.nn as nn

        state_dict_path = os.path.join(transform_dir, "mlp_transform.pt")
        state_dict = torch.load(state_dict_path, map_location="cpu")

        # Reconstruct the network
        net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        net.load_state_dict(state_dict)
        net.eval()

        class MLTransformWrapper:
            def __init__(self, net):
                self.net = net
                self.device = next(net.parameters()).device

            def transform(self, z_inf):
                self.net.eval()
                with torch.no_grad():
                    z_inf_t = torch.from_numpy(z_inf.astype(np.float32)).to(self.device)
                    z_pred = self.net(z_inf_t)
                return z_pred.cpu().numpy().astype(np.float32)

        return MLTransformWrapper(net), "mlp"

    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


class LinearTransform:
    """Simple linear transform: z_form = W @ z_inf + b"""

    def __init__(self, input_dim=768):
        self.W = None
        self.b = None
        self.input_dim = input_dim

    def fit(self, z_inf, z_form):
        """Fit linear regression: for each dimension, learn W[:,d] and b[d]."""
        n_samples = z_inf.shape[0]
        self.W = np.zeros((self.input_dim, z_form.shape[1]), dtype=np.float32)
        self.b = np.zeros(z_form.shape[1], dtype=np.float32)

        for d in range(z_form.shape[1]):
            reg = __import__('sklearn.linear_model').linear_model.LinearRegression().fit(z_inf, z_form[:, d])
            self.W[:, d] = reg.coef_.astype(np.float32)
            self.b[d] = reg.intercept_.astype(np.float32)

    def transform(self, z_inf):
        """Apply transform: z_form approx W @ z_inf + b"""
        return (z_inf @ self.W + self.b.astype(np.float32)).astype(np.float32)


class WhiteningProcrustesTransform:
    """Whitening + Orthogonal Procrustes transform."""

    def __init__(self):
        self.W = None
        self.mean_inf = None
        self.mean_form = None
        self.whitening_matrix = None

    def fit(self, z_inf, z_form):
        """Fit whitening + Procrustes transform."""
        n_samples = z_inf.shape[0]
        self.mean_inf = np.mean(z_inf, axis=0, keepdims=True)
        self.mean_form = np.mean(z_form, axis=0, keepdims=True)
        inf_centered = z_inf - self.mean_inf
        form_centered = z_form - self.mean_form
        cov = np.cov(inf_centered.T)
        U, S, Vt = np.linalg.svd(cov + 1e-6 * np.eye(cov.shape[0]))
        self.whitening_matrix = (U @ np.diag(1.0 / np.sqrt(S + 1e-5)) @ U.T)
        inf_whitened = inf_centered @ self.whitening_matrix.T
        M = form_centered.T @ inf_whitened
        U_p, _, Vt_p = np.linalg.svd(M)
        self.W = (U_p @ Vt_p).astype(np.float32)

    def transform(self, z_inf):
        """Apply transform: z_form approx W @ (z_inf - mean_inf) @ whitening.T + mean_form"""
        inf_centered = z_inf - self.mean_inf
        inf_whitened = inf_centered @ self.whitening_matrix.T
        return ((inf_whitened @ self.W.T) + self.mean_form).astype(np.float32)




def main():
    ap = argparse.ArgumentParser(description="Evaluate transformed embeddings for informal query retrieval")
    ap.add_argument("--transform_type", choices=["linear", "whitening_procrustes", "mlp"], default="linear",
                    help="Type of transform to evaluate")
    ap.add_argument("--transform_dir", default=os.path.join(HERE, "outputs", "transforms", "linear"),
                    help="Directory containing the trained transform")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"],
                    help="Evaluation split")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"), help="Data directory")
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "methods", "cache"), help="Cache directory with corpus embeddings")
    ap.add_argument("--max_length", type=int, default=64, help="BGE-M3 max sequence length")
    ap.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for significance")
    ap.add_argument("--output", default=None, help="Output JSON file path")
    args = ap.parse_args()

    root = ROOT  # Shortcut

    print(f"Evaluating {args.transform_type} transform on {args.split} split...")
    print(f"  Transform dir: {args.transform_dir}")
    print(f"  Split: {args.split}")

    # Load transform
    try:
        transform, ttype = load_transform(args.transform_dir, args.transform_type, root)
        print(f"  Loaded {ttype} transform")
    except Exception as e:
        print(f"  Error loading transform: {e}")
        print("  Please train a transform first using: python3 models/embed_transform/train_transform.py")
        return

    # Initialize encoder and embedding store
    encoder = BGEM3Encoder("BAAI/bge-m3")
    store = EmbeddingStore(args.cache_dir)

    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        splits = json.load(f)

    qids = splits[args.split]
    print(f"  Evaluating on {len(qids)} queries (split: {args.split})")

    # Extract informal query embeddings
    # We need to get the informal query texts for the qids in this split
    # Use the eval framework's approach

    # Load training pairs to map qids to informal queries
    train_pairs_df = load_train_pairs(os.path.join(root, "data"))
    n_pairs = len(train_pairs_df)

    # Build qid -> informal query mapping
    # The pairs are (formal, informal), but we need qid mapping
    # Let's check what qid format the dataset uses

    # From evaluate.py, we know ev.qids gives us the query IDs
    # And ev.informal() gives us the informal query embeddings
    # Let's follow the same pattern

    # Actually, let's extract the informal queries properly
    # The dataset has informal_query column

    pairs_df = pd.read_csv(os.path.join(root, "data", "train_pairs.csv"))
    pairs_df["formal"] = pairs_df["formal"].astype(str).str.strip()
    pairs_df["informal"] = pairs_df["informal"].astype(str).str.strip()

    # Build mapping from qid to query text
    # We need to know how qids relate to the pairs
    # Let's assume qids in the split correspond to rows in the pairs
    # Or we need to load the test/dev query IDs and their query texts

    # Simpler approach: use the existing evaluation infrastructure
    # Create evaluator and get the informal embeddings

    ev = Evaluator(store, json.load(open(os.path.join(args.cache_dir, "splits.json")))[args.split], args.data_dir)

    # Get informal query embeddings from the evaluator
    # This uses the base BGE-M3 encoder on informal queries
    z_inf_base = ev.informal()  # Shape: (n_queries, hidden_dim)
    print(f"  Extracted base informal embeddings: {z_inf_base.shape}")

    # Apply the transform
    print(f"  Applying {ttype} transform...")
    t0 = time.time()
    # Transform expects numpy array
    z_inf_numpy = z_inf_base.cpu().numpy()
    z_inf_transformed = transform.transform(z_inf_numpy)
    transform_time = time.time() - t0
    print(f"  Transform applied in {transform_time:.3f}s")
    print(f"  Transformed embeddings shape: {z_inf_transformed.shape}")

    # Get formal query embeddings (upper bound)
    z_formal = ev.formal()
    print(f"  Formal embeddings shape: {z_formal.shape}")

    # Run evaluation with transformed embeddings
    # We need to compute per-query metrics manually since we can't modify the store
    # Convert transformed embeddings to tensor for evaluation
    z_inf_transformed_tensor = torch.from_numpy(z_inf_transformed).to(ev.store.device)

    # Compute per-query nDCG@10, MRR@10, Recall@100 for transformed informal
    per_query = ev.run_per_query(z_inf_transformed_tensor)

    # Also get formal per-query for reference
    z_formal_tensor = ev.formal()
    z_formal_per_query = ev.run_per_query(z_formal_tensor)

    # Compute metrics
    results = {}

    # Base: formal queries
    formal_metrics = {}
    for m in METRICS:
        formal_metrics[m] = float(np.mean(z_formal_per_query[m]))
    results["formal"] = formal_metrics

    # Zero-shot: base informal queries
    z_inf_base = ev.informal()
    informal_per_query = ev.run_per_query(z_inf_base)
    zero_shot_metrics = {}
    for m in METRICS:
        zero_shot_metrics[m] = float(np.mean(informal_per_query[m]))
    results["zero_shot_informal"] = zero_shot_metrics

    # Transformed informal
    transformed_metrics = {}
    for m in METRICS:
        transformed_metrics[m] = float(np.mean(per_query[m]))
    results["transformed_informal"] = transformed_metrics

    # Cosine similarity between transformed and true formal
    z_formal_numpy = z_formal.cpu().numpy()
    z_inf_transformed_for_sim = z_inf_transformed  # Already numpy from transform
    sim_matrix = np.dot(z_inf_transformed_for_sim, z_formal_numpy.T) / (
        np.linalg.norm(z_inf_transformed_for_sim, axis=1, keepdims=True) *
        np.linalg.norm(z_formal_numpy, axis=1, keepdims=True) + 1e-8
    )
    avg_cos_sim = np.mean(np.diag(sim_matrix))
    print(f"  Average cosine similarity (transformed ≈ true formal): {avg_cos_sim:.4f}")
    results["avg_cosine_similarity"] = float(avg_cos_sim)

    # Gap closing calculation
    formal_nDCG = results["formal"]["nDCG@10"]
    transformed_nDCG = results["transformed_informal"]["nDCG@10"]
    zero_shot_nDCG = results.get("zero_shot_informal", {}).get("nDCG@10", transformed_nDCG)

    gap_closed = 100 * (transformed_nDCG - zero_shot_nDCG) / (formal_nDCG - zero_shot_nDCG) if (formal_nDCG != zero_shot_nDCG) else 0
    results["gap_closed_pct"] = float(max(0, min(100, gap_closed)))

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({args.split} split)")
    print(f"{'='*60}")
    print(f"{'Condition':<25} {'nDCG@10':>10} {'MRR@10':>10} {'Recall@100':>10}")
    print(f"{'Formal (upper bound)':<25} {results['formal']['nDCG@10']:>10.4f} {results['formal']['MRR@10']:>10.4f} {results['formal']['Recall@100']:>10.4f}")

    # Zero-shot informal
    if "zero_shot_informal" in results and results["zero_shot_informal"]["nDCG@10"] != "N/A":
        print(f"{'Zero-shot informal':<25} {results['zero_shot_informal']['nDCG@10']:>10.4f} {results['zero_shot_informal']['MRR@10']:>10.4f} {results['zero_shot_informal']['Recall@100']:>10.4f}")

    # Transformed informal
    print(f"{'Transformed informal':<25} {results['transformed_informal']['nDCG@10']:>10.4f} {results['transformed_informal']['MRR@10']:>10.4f} {results['transformed_informal']['Recall@100']:>10.4f}")

    print(f"{'Gap closed':<25} {results['gap_closed_pct']:>10.1f}%")

    # Compare to LLM-Norm baseline if available
    # LLM-Norm achieved ~81.2% gap closed on test split
    print(f"\n  LLM-Norm baseline: ~81.2% gap closed (for reference)")
    print(f"  This transform: {results['gap_closed_pct']:.1f}% gap closed")
    print(f"  vs TSDAE+Meaning: 28.0% gap closed (from earlier experiments)")

    # Save results if output path specified
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        results["split"] = args.split
        results["transform_type"] = ttype
        results["transform_dir"] = args.transform_dir
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()