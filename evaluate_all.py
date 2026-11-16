#!/usr/bin/env python3
"""Evaluate all methods on test split and produce comparison tables."""

import json
import os
import sys
import torch
import numpy as np

ROOT = "/home/yd/adapt-dann-retrieve/bge_dann"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "methods"))
sys.path.insert(0, os.path.join(ROOT, "methods", "query_distillation"))

from preprocessing.dataset import EmbeddingStore, load_train_pairs
from methods.eval import Evaluator, METRICS
from methods.model import BGEM3Encoder, ADAPTERS, QueryDANN, EMB_DIM
from methods.query_distillation.evaluate import load_adapter, adapt


def evaluate_base_model(cache_dir, data_dir, split="test"):
    """Evaluate base BGE-M3 on formal and informal queries."""
    store = EmbeddingStore(cache_dir)
    with open(os.path.join(cache_dir, "splits.json")) as f:
        splits = json.load(f)
    ev = Evaluator(store, splits[split], data_dir)
    
    # Base model on formal queries (upper bound)
    formal_results = ev.run(ev.formal())
    # Base model on informal queries (zero-shot)
    informal_results = ev.run(ev.informal())
    
    return formal_results, informal_results


def evaluate_query_dann(adapter_path, cache_dir, data_dir, split="test"):
    """Evaluate Query-DANN adapter."""
    store = EmbeddingStore(cache_dir)
    with open(os.path.join(cache_dir, "splits.json")) as f:
        splits = json.load(f)
    ev = Evaluator(store, splits[split], data_dir)
    
    # Load adapter
    adapter = load_adapter(adapter_path, store.device)
    
    # Adapt informal queries
    z_inf = ev.informal()
    z_adapted = adapt(adapter, z_inf)
    informal_results = ev.run(z_adapted)
    
    # Adapt formal queries (for comparison)
    z_form = ev.formal()
    z_form_adapted = adapt(adapter, z_form)
    formal_results = ev.run(z_form_adapted)
    
    return formal_results, informal_results


def evaluate_query_distillation(adapter_path, cache_dir, data_dir, split="test"):
    """Evaluate Query Distillation adapter."""
    return evaluate_query_dann(adapter_path, cache_dir, data_dir, split)


def evaluate_tsdae(model_path, cache_dir, data_dir, split="test"):
    """Evaluate TSDAE fine-tuned encoder."""
    store = EmbeddingStore(cache_dir)
    with open(os.path.join(cache_dir, "splits.json")) as f:
        splits = json.load(f)
    ev = Evaluator(store, splits[split], data_dir)
    
    # Load fine-tuned encoder
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path).to(store.device)
    
    # Get query texts
    tp = load_train_pairs(data_dir).set_index("query_id")
    
    # Encode informal queries
    informal_texts = [tp.at[q, "generated_informal_query"] for q in ev.qids]
    z_inf = model.encode(informal_texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)
    informal_results = ev.run(z_inf)
    
    # Encode formal queries
    formal_texts = [tp.at[q, "formal_query"] for q in ev.qids]
    z_form = model.encode(formal_texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)
    formal_results = ev.run(z_form)
    
    return formal_results, informal_results


def evaluate_query_infonce(model_path, cache_dir, data_dir, split="test"):
    """Evaluate Query-InfoNCE fine-tuned encoder."""
    return evaluate_tsdae(model_path, cache_dir, data_dir, split)


def evaluate_llm_rewrite(rewrites_path, cache_dir, data_dir, split="test"):
    """Evaluate LLM rewrite baseline."""
    store = EmbeddingStore(cache_dir)
    with open(os.path.join(cache_dir, "splits.json")) as f:
        splits = json.load(f)
    ev = Evaluator(store, splits[split], data_dir)
    
    with open(rewrites_path) as f:
        rows = {r["query_id"]: r for r in map(json.loads, filter(str.strip, f))}
    
    missing = [q for q in ev.qids if q not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} evaluated queries have no rewrite")
    
    encoder = BGEM3Encoder("BAAI/bge-m3")
    texts = [rows[q]["normalized"] for q in ev.qids]
    z = torch.from_numpy(encoder.encode(texts, max_length=64, desc="rewritten queries")).to(store.device).float()
    del encoder
    torch.cuda.empty_cache()
    
    rewrite_results = ev.run(z)
    
    return rewrite_results


def format_table(results_dict):
    """Format results as a comparison table."""
    # Header
    print(f"\n{'Model':<35} | {'Informal nDCG@10':>16} | {'Informal MRR@10':>16} | {'Informal R@100':>15} | {'Formal nDCG@10':>15} | {'Formal MRR@10':>15} | {'Formal R@100':>14}")
    print("-" * 155)
    
    for model_name, (formal, informal) in results_dict.items():
        print(f"{model_name:<35} | {informal['nDCG@10']:>16.4f} | {informal['MRR@10']:>16.4f} | {informal['Recall@100']:>15.4f} | {formal['nDCG@10']:>15.4f} | {formal['MRR@10']:>15.4f} | {formal['Recall@100']:>14.4f}")


def format_ndcg_table(results_dict):
    """Format nDCG@10 only comparison table."""
    print(f"\n{'Model':<35} | {'Informal nDCG@10':>18} | {'Formal nDCG@10':>17}")
    print("-" * 75)
    
    for model_name, (formal, informal) in results_dict.items():
        print(f"{model_name:<35} | {informal['nDCG@10']:>18.4f} | {formal['nDCG@10']:>17.4f}")


def main():
    cache_dir = os.path.join(ROOT, "methods", "cache")
    data_dir = os.path.join(ROOT, "data")
    split = "dev"
    
    print(f"Evaluating on {split} split...")
    print("=" * 80)
    
    results = {}
    
    # 1. Base BGE-M3
    print("\n1. Evaluating Base BGE-M3...")
    formal_base, informal_base = evaluate_base_model(cache_dir, data_dir, split)
    results["BGE-M3 (base)"] = (formal_base, informal_base)
    print(f"   Formal: nDCG@10={formal_base['nDCG@10']:.4f}, MRR@10={formal_base['MRR@10']:.4f}, R@100={formal_base['Recall@100']:.4f}")
    print(f"   Informal: nDCG@10={informal_base['nDCG@10']:.4f}, MRR@10={informal_base['MRR@10']:.4f}, R@100={informal_base['Recall@100']:.4f}")
    
    # 2. Query-DANN
    print("\n2. Evaluating Query-DANN...")
    dann_path = os.path.join(ROOT, "methods", "query_dann", "outputs", "query_dann", "best")
    formal_dann, informal_dann = evaluate_query_dann(dann_path, cache_dir, data_dir, split)
    results["Query-DANN"] = (formal_dann, informal_dann)
    print(f"   Formal: nDCG@10={formal_dann['nDCG@10']:.4f}, MRR@10={formal_dann['MRR@10']:.4f}, R@100={formal_dann['Recall@100']:.4f}")
    print(f"   Informal: nDCG@10={informal_dann['nDCG@10']:.4f}, MRR@10={informal_dann['MRR@10']:.4f}, R@100={informal_dann['Recall@100']:.4f}")
    
    # 3. Query Distillation (Score)
    print("\n3. Evaluating Query Distillation (Score)...")
    distill_path = os.path.join(ROOT, "methods", "query_distillation", "outputs", "harmonized", "best")
    formal_distill, informal_distill = evaluate_query_distillation(distill_path, cache_dir, data_dir, split)
    results["Query Distillation (Score)"] = (formal_distill, informal_distill)
    print(f"   Formal: nDCG@10={formal_distill['nDCG@10']:.4f}, MRR@10={formal_distill['MRR@10']:.4f}, R@100={formal_distill['Recall@100']:.4f}")
    print(f"   Informal: nDCG@10={informal_distill['nDCG@10']:.4f}, MRR@10={informal_distill['MRR@10']:.4f}, R@100={informal_distill['Recall@100']:.4f}")
    
    # 4. TSDAE (newly retrained)
    print("\n4. Evaluating TSDAE...")
    tsdae_path = os.path.join(ROOT, "methods", "tsdae_sapt_queries", "outputs", "tsdae_informal", "model")
    formal_tsdae, informal_tsdae = evaluate_tsdae(tsdae_path, cache_dir, data_dir, split)
    results["TSDAE-SAPT"] = (formal_tsdae, informal_tsdae)
    print(f"   Formal: nDCG@10={formal_tsdae['nDCG@10']:.4f}, MRR@10={formal_tsdae['MRR@10']:.4f}, R@100={formal_tsdae['Recall@100']:.4f}")
    print(f"   Informal: nDCG@10={informal_tsdae['nDCG@10']:.4f}, MRR@10={informal_tsdae['MRR@10']:.4f}, R@100={informal_tsdae['Recall@100']:.4f}")

    # 5. Query-InfoNCE (newly retrained)
    print("\n5. Evaluating Query-InfoNCE...")
    infonce_path = os.path.join(ROOT, "methods", "query_infonce", "outputs", "bge-m3-infonce-id")
    formal_infonce, informal_infonce = evaluate_query_infonce(infonce_path, cache_dir, data_dir, split)
    results["Query-InfoNCE"] = (formal_infonce, informal_infonce)
    print(f"   Formal: nDCG@10={formal_infonce['nDCG@10']:.4f}, MRR@10={formal_infonce['MRR@10']:.4f}, R@100={formal_infonce['Recall@100']:.4f}")
    print(f"   Informal: nDCG@10={informal_infonce['nDCG@10']:.4f}, MRR@10={informal_infonce['MRR@10']:.4f}, R@100={informal_infonce['Recall@100']:.4f}")
    
    # 6. LLM Rewrite (Gemini)
    print("\n6. Evaluating LLM Rewrite (Gemini)...")
    rewrites_path = os.path.join(ROOT, "methods", "llm_rewrite", "cache", "gemini-3.5-flash-lite_dev.jsonl")
    rewrite_results = evaluate_llm_rewrite(rewrites_path, cache_dir, data_dir, split)
    results["LLM Rewrite (Gemini)"] = (formal_base, rewrite_results)

    # 7. LLM Rewrite (Ollama Qwen2.5 3B)
    print("\n7. Evaluating LLM Rewrite (Ollama Qwen2.5 3B)...")
    rewrites_path_ollama = os.path.join(ROOT, "methods", "llm_rewrite", "cache", "qwen2.5_3b_instruct_q4_K_M_dev.jsonl")
    rewrite_results_ollama = evaluate_llm_rewrite(rewrites_path_ollama, cache_dir, data_dir, split)
    # For LLM rewrite, formal is the same as base formal (it's just a different query encoding)
    results["LLM Rewrite (Gemini)"] = (formal_base, rewrite_results)
    results["LLM Rewrite (Ollama Qwen2.5 3B)"] = (formal_base, rewrite_results_ollama)
    print(f"   Formal: nDCG@10={formal_base['nDCG@10']:.4f}, MRR@10={formal_base['MRR@10']:.4f}, R@100={formal_base['Recall@100']:.4f}")
    print(f"   Informal (rewritten Gemini): nDCG@10={rewrite_results['nDCG@10']:.4f}, MRR@10={rewrite_results['MRR@10']:.4f}, R@100={rewrite_results['Recall@100']:.4f}")
    print(f"   Informal (rewritten Ollama): nDCG@10={rewrite_results_ollama['nDCG@10']:.4f}, MRR@10={rewrite_results_ollama['MRR@10']:.4f}, R@100={rewrite_results_ollama['Recall@100']:.4f}")
    
    # Print comparison tables
    print("\n" + "=" * 80)
    print("COMPARISON TABLE: All Metrics")
    print("=" * 80)
    format_table(results)
    
    print("\n" + "=" * 80)
    print("COMPARISON TABLE: nDCG@10 Only")
    print("=" * 80)
    format_ndcg_table(results)
    
    # Calculate gap closed for nDCG@10
    print("\n" + "=" * 80)
    print("GAP CLOSED ANALYSIS (nDCG@10)")
    print("=" * 80)
    upper = formal_base['nDCG@10']
    lower = informal_base['nDCG@10']
    print(f"Upper bound (Formal): {upper:.4f}")
    print(f"Lower bound (Informal): {lower:.4f}")
    print(f"Gap: {upper - lower:.4f}")
    print()
    print(f"{'Model':<35} | {'Informal nDCG@10':>18} | {'Gap Closed':>12}")
    print("-" * 70)
    for model_name, (formal, informal) in results.items():
        if model_name == "BGE-M3 (base)":
            continue
        gap_closed = 100 * (informal['nDCG@10'] - lower) / (upper - lower)
        print(f"{model_name:<35} | {informal['nDCG@10']:>18.4f} | {gap_closed:>11.1f}%")
    
    # Save results to JSON
    output = {
        "split": split,
        "results": {
            model: {
                "formal": {k: float(v) for k, v in formal.items()},
                "informal": {k: float(v) for k, v in informal.items()}
            }
            for model, (formal, informal) in results.items()
        }
    }
    out_path = os.path.join(ROOT, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()