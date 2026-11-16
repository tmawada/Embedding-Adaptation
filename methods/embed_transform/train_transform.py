"""Train a learned transform to map informal query embeddings to formal query embeddings.

This script:
1. Loads (formal, informal) pairs from data/train_pairs.csv
2. Extracts embeddings using the frozen BGE-M3 encoder
3. Learns a transform T minimizing ||T(z_inf) - z_form||^2
4. Saves the transform for later use in retrieval

The transform is deliberately lightweight:
- Option A: Linear transform (W @ z + b) - 768x768 matrix + bias
- Option B: Linear + ReLU - small MLP
- Option C: Whitening + Procrustes (no learned params, better generalization)

Usage:
    python3 models/embed_transform/train_transform.py          # Default options
    python3 models/embed_transform/train_transform.py --linear  # Force linear transform
    python3 models/embed_transform/train_transform.py --mlp      # Force MLP transform
    python3 models/embed_transform/train_transform.py --whitening  # Whitening + Procrustes
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
from sklearn.linear_model import LinearRegression

# Add root to path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from methods.model import BGEM3Encoder  # noqa: E402
from preprocessing.dataset import load_train_pairs  # noqa: E402


# --------------------------------------------------------------------------- #
# Embedding Extraction
# --------------------------------------------------------------------------- #

def extract_embeddings(encoder, texts, max_length=64, batch_size=32, device="cuda"):
    """Extract BGE-M3 [CLS] embeddings for a list of query texts."""
    all_embeds = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # BGEM3Encoder.encode returns a numpy array of embeddings
        emb = encoder.encode(batch_texts, max_length=max_length)
        # The embeddings are already [CLS] pooled and L2 normalized
        all_embeds.append(emb)

    return np.vstack(all_embeds)


# --------------------------------------------------------------------------- #
# Transform Options
# --------------------------------------------------------------------------- #

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
            reg = LinearRegression().fit(z_inf, z_form[:, d])
            self.W[:, d] = reg.coef_.astype(np.float32)
            self.b[d] = reg.intercept_.astype(np.float32)
    
    def transform(self, z_inf):
        """Apply transform: z_form approx W @ z_inf + b"""
        return (z_inf @ self.W + self.b.astype(np.float32)).astype(np.float32)


class WhiteningProcrustesTransform:
    """Whitening + Orthogonal Procrustes transform.
    
    No learned parameters - better generalization to unseen data.
    Solves: min ||W * Z_inf_whitened - Z_form_centered||_F^2 s.t. W^T W = I
    """
    
    def __init__(self):
        self.W = None
        self.mean_inf = None
        mean_form = None
        self.whitening_matrix = None
        self.mean_form = None
    
    def fit(self, z_inf, z_form):
        """Fit whitening + Procrustes transform."""
        n_samples = z_inf.shape[0]
        
        # Center the data
        self.mean_inf = np.mean(z_inf, axis=0, keepdims=True)
        self.mean_form = np.mean(z_form, axis=0, keepdims=True)
        
        inf_centered = z_inf - self.mean_inf
        form_centered = z_form - self.mean_form
        
        # Whiten informal embeddings
        cov = np.cov(inf_centered.T)
        U, S, Vt = np.linalg.svd(cov + 1e-6 * np.eye(cov.shape[0]))
        self.whitening_matrix = (U @ np.diag(1.0 / np.sqrt(S + 1e-5)) @ U.T)
        
        inf_whitened = inf_centered @ self.whitening_matrix.T
        
        # Solve orthogonal Procrustes: min ||W @ inf_whitened - form_centered||_F^2 s.t. W^T W = I
        M = form_centered.T @ inf_whitened
        U_p, _, Vt_p = np.linalg.svd(M)
        self.W = (U_p @ Vt_p).astype(np.float32)
    
    def transform(self, z_inf):
        """Apply transform: z_form approx W @ (z_inf - mean_inf) @ whitening.T + mean_form"""
        inf_centered = z_inf - self.mean_inf
        inf_whitened = inf_centered @ self.whitening_matrix.T
        return ((inf_whitened @ self.W.T) + self.mean_form).astype(np.float32)


class MLTransform:
    """Small MLP transform: Linear -> ReLU -> Linear"""
    
    def __init__(self, input_dim=768, hidden_dim=256):
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
    
    def fit(self, z_inf, z_form, epochs=10, lr=1e-3, batch_size=32):
        """Fit MLP transform using PyTorch."""
        z_inf_t = torch.from_numpy(z_inf.astype(np.float32)).to(self.device)
        z_form_t = torch.from_numpy(z_form.astype(np.float32)).to(self.device)
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(z_inf_t, z_form_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.net.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for z_inf_batch, z_form_batch in loader:
                optimizer.zero_grad()
                z_pred = self.net(z_inf_batch)
                loss = criterion(z_pred, z_form_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            if (epoch + 1) % 5 == 0 or epoch == 1:
                print(f"  MLP epoch {epoch+1}/{epochs}, loss: {avg_loss:.6f}")
        
        self.net.eval()
    
    def transform(self, z_inf):
        """Apply transform: z_form approx net(z_inf)"""
        self.net.eval()
        with torch.no_grad():
            z_inf_t = torch.from_numpy(z_inf.astype(np.float32)).to(self.device)
            z_pred = self.net(z_inf_t)
        return z_pred.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Evaluation Helper
# --------------------------------------------------------------------------- #

def cosine_similarity_matrix(z_inf, z_form):
    """Compute cosine similarity matrix between two sets of vectors."""
    z_inf_norm = z_inf / (np.linalg.norm(z_inf, axis=1, keepdims=True) + 1e-8)
    z_form_norm = z_form / (np.linalg.norm(z_form, axis=1, keepdims=True) + 1e-8)
    return z_inf_norm @ z_form_norm.T


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Train embedding transform for informal->formal alignment")
    ap.add_argument("--linear", action="store_true", help="Use linear transform (default if no --mlp/--whitening)")
    ap.add_argument("--mlp", action="store_true", help="Use MLP transform with hidden layer")
    ap.add_argument("--whitening", action="store_true", help="Use whitening + Procrustes transform")
    ap.add_argument("--epochs", type=int, default=10, help="MLP training epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="MLP learning rate")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding extraction and training")
    ap.add_argument("--max_length", type=int, default=64, help="BGE-M3 max sequence length")
    ap.add_argument("--output_dir", default=os.path.join(HERE, "outputs", "transforms"), help="Output directory for saved transforms")
    ap.add_argument("--eval_split", default="test", choices=["train", "dev", "test"], help="Evaluation split for reporting similarity")
    ap.add_argument("--train_split_only", action="store_true",
                    help="Fit on the train-split queries only (splits.json in --cache_dir); default: every pair in "
                         "train_pairs.csv, which includes the dev and test queries")
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "methods", "cache"), help="Holds splits.json")
    args = ap.parse_args()
    
    # Determine transform type
    if args.mlp:
        transform_type = "mlp"
    elif args.whitening:
        transform_type = "whitening_procrustes"
    else:
        transform_type = "linear"  # Default: linear
    
    print(f"Training {transform_type} transform...")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Max length: {args.max_length}")
    print(f"  Batch size: {args.batch_size}")
    
    # Load training pairs
    data_dir = "/home/yd/adapt-dann-retrieve/bge_dann/data"
    df = load_train_pairs(data_dir)
    if args.train_split_only:
        with open(os.path.join(args.cache_dir, "splits.json")) as f:
            df = df[df["query_id"].isin(set(json.load(f)["train"]))]
    pairs = list(zip(df["formal_query"], df["generated_informal_query"]))
    n_pairs = len(pairs)
    print(f"  Pairs loaded: {n_pairs}")
    
    # Initialize encoder (frozen)
    encoder = BGEM3Encoder("BAAI/bge-m3")
    device = encoder.device
    print(f"  Using device: {device}")
    
    # Extract embeddings
    print("  Extracting formal query embeddings...")
    formal_texts = [pair[0] for pair in pairs]
    z_form = extract_embeddings(encoder, formal_texts, max_length=args.max_length, batch_size=args.batch_size)
    print(f"    Shape: {z_form.shape} (n_queries x embedding_dim)")
    
    print("  Extracting informal query embeddings...")
    informal_texts = [pair[1] for pair in pairs]
    z_inf = extract_embeddings(encoder, informal_texts, max_length=args.max_length, batch_size=args.batch_size)
    print(f"    Shape: {z_inf.shape} (n_queries x embedding_dim)")
    
    # Learn transform
    print(f"\n  Training {transform_type} transform...")
    transform = None
    
    if transform_type == "linear":
        transform = LinearTransform(input_dim=z_inf.shape[1])
        t0 = time.time()
        transform.fit(z_inf, z_form)
        print(f"  Training time: {time.time() - t0:.2f}s")
        
        # Compute training similarity
        z_form_approx = transform.transform(z_inf)
        sim_matrix = cosine_similarity_matrix(z_form_approx, z_form)
        avg_sim = np.mean(np.diag(sim_matrix))
        print(f"  Average cosine similarity (diagonal): {avg_sim:.4f}")
        print(f"  Min diag sim: {np.min(np.diag(sim_matrix)):.4f}")
        print(f"  Max diag sim: {np.max(np.diag(sim_matrix)):.4f}")
    
    elif transform_type == "whitening_procrustes":
        transform = WhiteningProcrustesTransform()
        t0 = time.time()
        transform.fit(z_inf, z_form)
        print(f"  Training time: {time.time() - t0:.2f}s")
        
        # Compute training similarity
        z_form_approx = transform.transform(z_inf)
        sim_matrix = cosine_similarity_matrix(z_form_approx, z_form)
        avg_sim = np.mean(np.diag(sim_matrix))
        print(f"  Average cosine similarity (diagonal): {avg_sim:.4f}")
        print(f"  Min diag sim: {np.min(np.diag(sim_matrix)):.4f}")
        print(f"  Max diag sim: {np.max(np.diag(sim_matrix)):.4f}")
    
    elif transform_type == "mlp":
        transform = MLTransform(input_dim=z_inf.shape[1])
        t0 = time.time()
        transform.fit(z_inf, z_form, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        print(f"  Training time: {time.time() - t0:.2f}s")
        
        # Compute training similarity
        z_form_approx = transform.transform(z_inf)
        sim_matrix = cosine_similarity_matrix(z_form_approx, z_form)
        avg_sim = np.mean(np.diag(sim_matrix))
        print(f"  Average cosine similarity (diagonal): {avg_sim:.4f}")
        print(f"  Min diag sim: {np.min(np.diag(sim_matrix)):.4f}")
        print(f"  Max diag sim: {np.max(np.diag(sim_matrix)):.4f}")
    
    # Save transform
    os.makedirs(args.output_dir, exist_ok=True)
    
    save_data = {
        "transform_type": transform_type,
        "input_dim": z_inf.shape[1],
        "z_inf_shape": z_inf.shape,
        "z_form_shape": z_form.shape,
        "transform_params": {},
    }
    
    if transform_type == "linear":
        save_data["transform_params"] = {
            "W": transform.W.tolist(),
            "b": transform.b.tolist(),
        }
        # Save as numpy for easy loading
        np.savez(os.path.join(args.output_dir, "linear_transform.npz"),
                 W=transform.W, b=transform.b,
                 z_inf_mean=None, z_inf_whitening=None,
                 z_form_mean=None)
    
    elif transform_type == "whitening_procrustes":
        save_data["transform_params"] = {
            "W": transform.W.tolist(),
            "mean_inf": transform.mean_inf.tolist() if transform.mean_inf is not None else None,
            "mean_form": transform.mean_form.tolist() if transform.mean_form is not None else None,
            "whitening_matrix": transform.whitening_matrix.tolist() if transform.whitening_matrix is not None else None,
        }
        np.savez(os.path.join(args.output_dir, "whitening_procrustes_transform.npz"),
                 W=transform.W,
                 mean_inf=transform.mean_inf if transform.mean_inf is not None else np.zeros(z_inf.shape[1]),
                 mean_form=transform.mean_form if transform.mean_form is not None else np.zeros(z_form.shape[1]),
                 whitening_matrix=transform.whitening_matrix if transform.whitening_matrix is not None else np.eye(z_inf.shape[1]))
    
    elif transform_type == "mlp":
        save_data["transform_params"] = {
            "state_dict_path": "mlp_transform.pt",
        }
        # Save full model
        torch.save(transform.net.state_dict(),
                   os.path.join(args.output_dir, "mlp_transform.pt"))
    
    with open(os.path.join(args.output_dir, "transform_config.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nTransform saved to: {args.output_dir}/")
    print(f"  Config: transform_config.json")
    if transform_type == "linear":
        print(f"  Weights: linear_transform.npz (W: {transform.W.shape}, b: {transform.b.shape})")
    elif transform_type == "whitening_procrustes":
        print(f"  Weights: whitening_procrustes_transform.npz")
    elif transform_type == "mlp":
        print(f"  Weights: mlp_transform.pt (PyTorch state dict)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
