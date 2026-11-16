#!/usr/bin/env python3
"""Unified benchmark runner for BGE-M3 vs Query-DANN efficiency comparison.

This script implements the benchmark specification from query_dann_benchmark_spec.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

# Add project root to path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
BENCHMARK_DIR = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, BENCHMARK_DIR)

from models.baseline import BaselineModel, load_baseline_model
from models.query_dann import QueryDANNModel, load_query_dann_model
from profilers.latency import LatencyProfiler, LatencyStats
from profilers.memory import MemoryProfiler, MemoryStats
from profilers.parameters import ParameterCounts, analyze_model_parameters, get_checkpoint_sizes
from profilers.system import SystemMetadata, collect_system_metadata, save_metadata, print_system_summary
from evaluation.retrieval import RetrievalEvaluator, RetrievalResult, run_retrieval_benchmark


@dataclass
class BenchmarkConfig:
    """Benchmark configuration from YAML."""
    experiment_name: str
    seed: int
    encoder: str
    modes: List[str]
    devices: List[str]
    batch_sizes: List[int]
    warmup_steps: int
    repetitions: int
    top_k: int
    precision: str
    max_length: int
    normalize_embeddings: bool
    output_dir: str
    save_raw_latency: bool
    save_memory_samples: bool
    save_hardware_metadata: bool


def load_config(config_path: str) -> BenchmarkConfig:
    """Load benchmark configuration from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    return BenchmarkConfig(
        experiment_name=cfg["experiment"]["name"],
        seed=cfg["experiment"]["seed"],
        encoder=cfg["models"]["encoder"],
        modes=cfg["benchmark"]["modes"],
        devices=cfg["benchmark"]["devices"],
        batch_sizes=cfg["benchmark"]["batch_sizes"],
        warmup_steps=cfg["benchmark"]["warmup_steps"],
        repetitions=cfg["benchmark"]["repetitions"],
        top_k=cfg["benchmark"]["top_k"],
        precision=cfg["runtime"]["precision"],
        max_length=cfg["runtime"]["max_length"],
        normalize_embeddings=cfg["runtime"]["normalize_embeddings"],
        output_dir=cfg["output"]["directory"],
        save_raw_latency=cfg["output"]["save_raw_latency"],
        save_memory_samples=cfg["output"]["save_memory_samples"],
        save_hardware_metadata=cfg["output"]["save_hardware_metadata"],
    )


@dataclass
class RunResult:
    """Results from a single benchmark run."""
    run_id: str
    timestamp: str
    model_condition: str  # M0, M1, M2
    device: str
    precision: str
    batch_size: int
    max_length: int
    mode: str  # model_only, full_pipeline
    warmup_steps: int
    repetitions: int

    # Latency (ms per query)
    encoder_latency_mean: float
    encoder_latency_median: float
    encoder_latency_p95: float
    encoder_latency_std: float
    encoder_throughput_qps: float

    adapter_latency_mean: float
    adapter_latency_median: float
    adapter_latency_p95: float
    adapter_latency_std: float
    adapter_throughput_qps: float

    total_latency_mean: float
    total_latency_median: float
    total_latency_p95: float
    total_latency_std: float
    total_throughput_qps: float

    # Retrieval metrics (for full_pipeline) - main comparison condition
    ndcg_at_10: Optional[float] = None
    mrr_at_10: Optional[float] = None
    recall_at_100: Optional[float] = None

    # Detailed retrieval metrics (all conditions)
    retrieval_detailed: Optional[Dict] = None

    # Memory
    peak_rss_mb: Optional[float] = None
    peak_vram_mb: Optional[float] = None

    # Metadata
    raw_latencies: Optional[List[float]] = None


def generate_sample_queries(num_queries: int = 100, max_length: int = 64) -> List[str]:
    """Generate sample queries for benchmarking."""
    # Use realistic Indonesian informal queries similar to MIRACL-id
    templates = [
        "siapa {entity} ya",
        "{entity} itu apa sih",
        "gimana caranya {action}",
        "kenapa {entity} {action}",
        "dimana {entity} biasanya {action}",
        "kapan {entity} {action}",
        "berapa {entity} harganya",
        "mana yang lebih {adjective} {entity1} atau {entity2}",
    ]

    entities = ["presiden indonesia", "gunung tertinggi", "ibukota jepang", "mata uang eropa", "pemilik tesla",
                "penemu lampu", "penulis harry potter", "film avatar", "lautan pasifik", "gunung everest"]
    actions = ["dibuat", "ditemukan", "ditulis", "diluncurkan", "dibangun", "dikenal", "terjadi"]
    adjectives = ["bagus", "murah", "cepat", "besar", "tinggi", "populer", "unik"]

    import random
    random.seed(42)
    queries = []
    for _ in range(num_queries):
        template = random.choice(templates)
        query = template.format(
            entity=random.choice(entities),
            entity1=random.choice(entities),
            entity2=random.choice(entities),
            action=random.choice(actions),
            adjective=random.choice(adjectives),
        )
        queries.append(query)

    return queries


def run_model_only_benchmark(
    model,
    model_condition: str,
    device: torch.device,
    config: BenchmarkConfig,
    batch_size: int,
    sample_queries: List[str],
    profiler: LatencyProfiler,
    memory_profiler: MemoryProfiler,
) -> Dict[str, LatencyStats]:
    """Run model-only benchmark (encoding + adapter)."""
    print(f"\n  Running model-only benchmark: {model_condition} on {device.type} (batch={batch_size})")

    # Prepare batched queries
    num_batches = (len(sample_queries) + batch_size - 1) // batch_size
    batched_queries = [
        sample_queries[i * batch_size:(i + 1) * batch_size]
        for i in range(num_batches)
    ]

    results = {}

    # 1. Encoder latency
    def encode_batch(queries):
        return model.encode(queries)

    # Use first batch for profiling
    test_batch = batched_queries[0]
    encoder_stats = profiler.profile_with_batch(encode_batch, batch_size, test_batch)
    results["encoder"] = encoder_stats
    print(f"    Encoder: median={encoder_stats.median_ms:.2f}ms, p95={encoder_stats.p95_ms:.2f}ms, throughput={encoder_stats.throughput_qps:.1f} q/s")

    # 2. Adapter latency (if applicable)
    if hasattr(model, 'adapt') and model_condition != "M0":
        # Get embeddings first
        test_embeddings = model.encode(test_batch)
        def adapt_batch(emb):
            return model.adapt(emb)

        adapter_stats = profiler.profile_with_batch(adapt_batch, batch_size, test_embeddings)
        results["adapter"] = adapter_stats
        print(f"    Adapter: median={adapter_stats.median_ms:.2f}ms, p95={adapter_stats.p95_ms:.2f}ms, throughput={adapter_stats.throughput_qps:.1f} q/s")

        # 3. Total (encode + adapt)
        def encode_and_adapt_batch(queries):
            emb = model.encode(queries)
            return model.adapt(emb)

        total_stats = profiler.profile_with_batch(encode_and_adapt_batch, batch_size, test_batch)
        results["total"] = total_stats
        print(f"    Total: median={total_stats.median_ms:.2f}ms, p95={total_stats.p95_ms:.2f}ms, throughput={total_stats.throughput_qps:.1f} q/s")
    else:
        # M0: no adapter
        results["adapter"] = LatencyStats(0, 0, 0, 0, 0, 0, 0, [])
        results["total"] = encoder_stats
        print(f"    Total (no adapter): median={encoder_stats.median_ms:.2f}ms, p95={encoder_stats.p95_ms:.2f}ms, throughput={encoder_stats.throughput_qps:.1f} q/s")

    return results


def run_full_pipeline_benchmark(
    model,
    model_condition: str,
    device: torch.device,
    config: BenchmarkConfig,
    batch_size: int,
    sample_queries: List[str],
    profiler: LatencyProfiler,
    memory_profiler: MemoryProfiler,
    retrieval_evaluator: RetrievalEvaluator,
) -> Dict:
    """Run full-pipeline benchmark (encoding + adapter + retrieval)."""
    print(f"\n  Running full-pipeline benchmark: {model_condition} on {device.type} (batch={batch_size})")

    results = {}

    # For full pipeline, we measure end-to-end latency for a batch
    test_batch = sample_queries[:batch_size]

    # 1. Encoder latency
    def encode_batch(queries):
        return model.encode(queries)

    encoder_stats = profiler.profile_with_batch(encode_batch, batch_size, test_batch)
    results["encoder_latency"] = encoder_stats
    print(f"    Encoder: median={encoder_stats.median_ms:.2f}ms")

    # 2. Adapter latency (if applicable)
    if hasattr(model, 'adapt') and model_condition != "M0":
        test_embeddings = model.encode(test_batch)
        def adapt_batch(emb):
            return model.adapt(emb)

        adapter_stats = profiler.profile_with_batch(adapt_batch, batch_size, test_embeddings)
        results["adapter_latency"] = adapter_stats
        print(f"    Adapter: median={adapter_stats.median_ms:.2f}ms")

        # 3. Retrieval latency
        adapted_emb = model.adapt(test_embeddings)

        def retrieve_batch(emb):
            # Corpus is on CPU, move query embeddings to CPU for retrieval
            return retrieval_evaluator.evaluator._ranked(emb.cpu(), config.top_k)

        retrieval_stats = profiler.profile_with_batch(retrieve_batch, batch_size, adapted_emb)
        results["retrieval_latency"] = retrieval_stats
        print(f"    Retrieval: median={retrieval_stats.median_ms:.2f}ms")

        # 4. Total pipeline
        def full_pipeline_batch(queries):
            emb = model.encode(queries)
            adapted = model.adapt(emb)
            return retrieval_evaluator.evaluator._ranked(adapted.cpu(), config.top_k)

        total_stats = profiler.profile_with_batch(full_pipeline_batch, batch_size, test_batch)
        results["total_latency"] = total_stats
        print(f"    Total pipeline: median={total_stats.median_ms:.2f}ms, throughput={total_stats.throughput_qps:.1f} q/s")
    else:
        # M0: no adapter
        results["adapter_latency"] = LatencyStats(0, 0, 0, 0, 0, 0, 0, [])
        test_embeddings = model.encode(test_batch)

        def retrieve_batch(emb):
            return retrieval_evaluator.evaluator._ranked(emb.cpu(), config.top_k)

        retrieval_stats = profiler.profile_with_batch(retrieve_batch, batch_size, test_embeddings)
        results["retrieval_latency"] = retrieval_stats
        print(f"    Retrieval: median={retrieval_stats.median_ms:.2f}ms")

        def full_pipeline_batch(queries):
            emb = model.encode(queries)
            return retrieval_evaluator.evaluator._ranked(emb.cpu(), config.top_k)

        total_stats = profiler.profile_with_batch(full_pipeline_batch, batch_size, test_batch)
        results["total_latency"] = total_stats
        print(f"    Total pipeline: median={total_stats.median_ms:.2f}ms, throughput={total_stats.throughput_qps:.1f} q/s")

    # 5. Retrieval effectiveness (use full evaluator)
    print(f"    Running retrieval evaluation...")
    retrieval_results = run_retrieval_benchmark(
        model, retrieval_evaluator, model_condition, device, batch_size
    )
    results["retrieval_metrics"] = retrieval_results

    return results


def save_results_csv(results: List[RunResult], output_path: str):
    """Save benchmark results to CSV."""
    if not results:
        return

    fieldnames = list(asdict(results[0]).keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            # Convert lists to JSON strings for CSV
            if row["raw_latencies"] is not None:
                row["raw_latencies"] = json.dumps(row["raw_latencies"])
            writer.writerow(row)


def save_results_json(results: List[RunResult], output_path: str):
    """Save benchmark results to JSON."""
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)


def save_detailed_retrieval(results: List[RunResult], output_path: str):
    """Save detailed retrieval results (formal, informal, adapted) to JSON."""
    detailed = {}
    for r in results:
        if r.mode != "full_pipeline" or not r.retrieval_detailed:
            continue
        key = f"{r.model_condition}_{r.device}_batch{r.batch_size}"
        detailed[key] = r.retrieval_detailed
    with open(output_path, "w") as f:
        json.dump(detailed, f, indent=2, default=str)


def create_run_result(
    run_id: str,
    model_condition: str,
    device: str,
    precision: str,
    batch_size: int,
    max_length: int,
    mode: str,
    warmup_steps: int,
    repetitions: int,
    latency_results: Dict,
    retrieval_metrics: Optional[Dict] = None,
    memory_stats: Optional[MemoryStats] = None,
) -> RunResult:
    """Create a RunResult from benchmark data."""
    # Extract latency stats
    encoder_stats = latency_results.get("encoder") or latency_results.get("encoder_latency")
    adapter_stats = latency_results.get("adapter") or latency_results.get("adapter_latency")
    total_stats = latency_results.get("total") or latency_results.get("total_latency")

    result = RunResult(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        model_condition=model_condition,
        device=device,
        precision=precision,
        batch_size=batch_size,
        max_length=max_length,
        mode=mode,
        warmup_steps=warmup_steps,
        repetitions=repetitions,
        encoder_latency_mean=encoder_stats.mean_ms if encoder_stats else 0,
        encoder_latency_median=encoder_stats.median_ms if encoder_stats else 0,
        encoder_latency_p95=encoder_stats.p95_ms if encoder_stats else 0,
        encoder_latency_std=encoder_stats.std_ms if encoder_stats else 0,
        encoder_throughput_qps=encoder_stats.throughput_qps if encoder_stats else 0,
        adapter_latency_mean=adapter_stats.mean_ms if adapter_stats else 0,
        adapter_latency_median=adapter_stats.median_ms if adapter_stats else 0,
        adapter_latency_p95=adapter_stats.p95_ms if adapter_stats else 0,
        adapter_latency_std=adapter_stats.std_ms if adapter_stats else 0,
        adapter_throughput_qps=adapter_stats.throughput_qps if adapter_stats else 0,
        total_latency_mean=total_stats.mean_ms if total_stats else 0,
        total_latency_median=total_stats.median_ms if total_stats else 0,
        total_latency_p95=total_stats.p95_ms if total_stats else 0,
        total_latency_std=total_stats.std_ms if total_stats else 0,
        total_throughput_qps=total_stats.throughput_qps if total_stats else 0,
        peak_rss_mb=memory_stats.process_rss_mb if memory_stats else None,
        peak_vram_mb=memory_stats.gpu_peak_allocated_mb if memory_stats and memory_stats.gpu_peak_allocated_mb else None,
        raw_latencies=total_stats.raw_latencies_ms if total_stats else None,
    )

    # Add retrieval metrics if available
    if retrieval_metrics:
        # Get the main condition (adapted_informal for M2, informal for M0/M1)
        key = f"{model_condition}_adapted_informal" if model_condition in ["M1", "M2"] else f"{model_condition}_informal"
        if key not in retrieval_metrics:
            key = f"{model_condition}_informal"
        if key in retrieval_metrics:
            metrics = retrieval_metrics[key]
            result.ndcg_at_10 = metrics.nDCG_at_10
            result.mrr_at_10 = metrics.MRR_at_10
            result.recall_at_100 = metrics.Recall_at_100
        
        # Store detailed retrieval results
        detailed = {}
        for cond_name, cond_metrics in retrieval_metrics.items():
            detailed[cond_name] = {
                "nDCG@10": cond_metrics.nDCG_at_10,
                "MRR@10": cond_metrics.MRR_at_10,
                "Recall@100": cond_metrics.Recall_at_100,
            }
        result.retrieval_detailed = detailed

    return result


def main():
    parser = argparse.ArgumentParser(description="BGE-M3 vs Query-DANN Benchmark")
    parser.add_argument("--config", default=os.path.join(BENCHMARK_DIR, "benchmark-test", "configs", "benchmark.yaml"),
                        help="Path to benchmark config YAML")
    parser.add_argument("--model", choices=["baseline", "query_dann", "all"], default="all",
                        help="Model condition to benchmark")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to Query-DANN checkpoint (for M2)")
    parser.add_argument("--device", choices=["cpu", "cuda", "all"], default="all",
                        help="Device to run on")
    parser.add_argument("--modes", nargs="+", choices=["model_only", "full_pipeline"], default=None,
                        help="Benchmark modes to run")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                        help="Batch sizes to test")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--run-id", default=None,
                        help="Run ID (auto-generated if not provided)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with reduced repetitions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.modes:
        config.modes = args.modes
    if args.batch_sizes:
        config.batch_sizes = args.batch_sizes
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.quick:
        config.warmup_steps = 5
        config.repetitions = 10

    # Generate run ID
    run_id = args.run_id or f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup output directory
    output_base = os.path.join(BENCHMARK_DIR, config.output_dir, run_id)
    os.makedirs(output_base, exist_ok=True)

    # Collect system metadata
    device_for_metadata = args.device if args.device != "all" else ("cuda" if torch.cuda.is_available() else "cpu")
    metadata = collect_system_metadata(
        run_id=run_id,
        device=device_for_metadata,
        precision=config.precision,
        max_length=config.max_length,
        batch_size=config.batch_sizes[0] if config.batch_sizes else 1,
    )
    print_system_summary(metadata)
    save_metadata(metadata, os.path.join(output_base, "hardware"))

    # Generate sample queries
    sample_queries = generate_sample_queries(num_queries=max(config.batch_sizes) * 10, max_length=config.max_length)
    print(f"\nGenerated {len(sample_queries)} sample queries for benchmarking")

    # Initialize retrieval evaluator for full_pipeline mode
    retrieval_evaluator = None
    if "full_pipeline" in config.modes:
        max_queries = 50 if args.quick else None
        retrieval_evaluator = RetrievalEvaluator(
            top_k=config.top_k,
            device=torch.device("cpu"),  # Embedding store on CPU for memory efficiency
            max_queries=max_queries,
        )

    # Determine devices to test
    devices_to_test = []
    if args.device == "all":
        devices_to_test.append("cpu")
        if torch.cuda.is_available():
            devices_to_test.append("cuda")
    else:
        devices_to_test.append(args.device)

    # Determine models to test
    models_to_test = []
    if args.model in ["baseline", "all"]:
        models_to_test.append(("M0", "baseline"))
    if args.model in ["query_dann", "all"]:
        models_to_test.append(("M1", "query_dann_untrained"))
        if args.checkpoint:
            models_to_test.append(("M2", "query_dann_trained"))

    all_results = []

    # Run benchmarks
    for model_condition, model_type in models_to_test:
        for device_str in devices_to_test:
            device = torch.device(device_str)

            # Skip CUDA if not available
            if device_str == "cuda" and not torch.cuda.is_available():
                print(f"\nSkipping {model_condition} on CUDA (not available)")
                continue

            print(f"\n{'='*60}")
            print(f"Benchmarking {model_condition} on {device_str.upper()}")
            print(f"{'='*60}")

            # Load model
            if model_type == "baseline":
                model = load_baseline_model(
                    model_name=config.encoder,
                    device=device_str,
                    max_length=config.max_length,
                    precision=config.precision,
                )
            elif model_type == "query_dann_untrained":
                model = load_query_dann_model(
                    model_name=config.encoder,
                    device=device_str,
                    max_length=config.max_length,
                    precision=config.precision,
                    checkpoint_path=None,  # M1: untrained
                )
            elif model_type == "query_dann_trained":
                model = load_query_dann_model(
                    model_name=config.encoder,
                    device=device_str,
                    max_length=config.max_length,
                    precision=config.precision,
                    checkpoint_path=args.checkpoint,
                )
            else:
                continue

            model.to(device).eval()

            # Initialize profilers
            profiler = LatencyProfiler(
                warmup_steps=config.warmup_steps,
                repetitions=config.repetitions,
                device=device,
            )
            memory_profiler = MemoryProfiler(device=device)

            # Profile parameter counts (once per model)
            if model_condition in ["M0"]:
                param_counts = analyze_model_parameters(model.encoder.model)
            else:
                param_counts = analyze_model_parameters(
                    model.encoder.model,
                    model.adapter,
                    model.discriminator,
                )

            # Save parameter counts
            param_output = os.path.join(output_base, "parameters", f"{model_condition}_params.json")
            os.makedirs(os.path.dirname(param_output), exist_ok=True)
            with open(param_output, "w") as f:
                json.dump({
                    "encoder_total": param_counts.encoder_total,
                    "encoder_trainable": param_counts.encoder_trainable,
                    "adapter_total": param_counts.adapter_total,
                    "adapter_trainable": param_counts.adapter_trainable,
                    "discriminator_total": param_counts.discriminator_total,
                    "discriminator_trainable": param_counts.discriminator_trainable,
                    "inference_m0": param_counts.inference_total_m0,
                    "inference_m1_m2": param_counts.inference_total_m1_m2,
                    "training_total": param_counts.training_total,
                    "adapter_ratio_pct": param_counts.adapter_to_encoder_ratio_pct,
                }, f, indent=2)

            for batch_size in config.batch_sizes:
                # Skip batch sizes that are too large for the sample
                if batch_size > len(sample_queries):
                    print(f"  Skipping batch_size={batch_size} (need at least {batch_size} queries)")
                    continue

                # Run model-only benchmark
                if "model_only" in config.modes:
                    latency_results = run_model_only_benchmark(
                        model, model_condition, device, config, batch_size,
                        sample_queries, profiler, memory_profiler
                    )

                    # Memory snapshot
                    memory_stats = memory_profiler.get_memory_stats()

                    # Create result
                    result = create_run_result(
                        run_id=run_id,
                        model_condition=model_condition,
                        device=device_str,
                        precision=config.precision,
                        batch_size=batch_size,
                        max_length=config.max_length,
                        mode="model_only",
                        warmup_steps=config.warmup_steps,
                        repetitions=config.repetitions,
                        latency_results=latency_results,
                        memory_stats=memory_stats,
                    )
                    all_results.append(result)

                # Run full-pipeline benchmark
                if "full_pipeline" in config.modes and retrieval_evaluator:
                    fp_results = run_full_pipeline_benchmark(
                        model, model_condition, device, config, batch_size,
                        sample_queries, profiler, memory_profiler, retrieval_evaluator
                    )

                    # Memory snapshot
                    memory_stats = memory_profiler.get_memory_stats()

                    # Create result
                    result = create_run_result(
                        run_id=run_id,
                        model_condition=model_condition,
                        device=device_str,
                        precision=config.precision,
                        batch_size=batch_size,
                        max_length=config.max_length,
                        mode="full_pipeline",
                        warmup_steps=config.warmup_steps,
                        repetitions=config.repetitions,
                        latency_results=fp_results,
                        retrieval_metrics=fp_results.get("retrieval_metrics"),
                        memory_stats=memory_stats,
                    )
                    all_results.append(result)

    # Save all results
    print(f"\n{'='*60}")
    print(f"Saving results to {output_base}")
    print(f"{'='*60}")

    # Save CSV
    csv_path = os.path.join(output_base, "summary", "benchmark_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    save_results_csv(all_results, csv_path)

    # Save JSON
    json_path = os.path.join(output_base, "summary", "benchmark_results.json")
    save_results_json(all_results, json_path)

    # Save detailed retrieval results
    retrieval_path = os.path.join(output_base, "full_pipeline", "retrieval_detailed.json")
    os.makedirs(os.path.dirname(retrieval_path), exist_ok=True)
    save_detailed_retrieval(all_results, retrieval_path)

    # Save raw latencies if requested
    if config.save_raw_latency:
        raw_path = os.path.join(output_base, "raw", "latency_observations.json")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        raw_data = []
        for r in all_results:
            if r.raw_latencies:
                raw_data.append({
                    "run_id": r.run_id,
                    "model_condition": r.model_condition,
                    "device": r.device,
                    "batch_size": r.batch_size,
                    "mode": r.mode,
                    "latencies_ms": r.raw_latencies,
                })
        with open(raw_path, "w") as f:
            json.dump(raw_data, f, indent=2)

    # Print summary table
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Model':<6} {'Device':<6} {'Batch':<6} {'Mode':<14} {'Encoder(ms)':<12} {'Adapter(ms)':<12} {'Total(ms)':<12} {'Throughput':<12} {'nDCG@10':<8}")
    print("-" * 100)
    for r in all_results:
        ndcg = f"{r.ndcg_at_10:.4f}" if r.ndcg_at_10 else "N/A"
        print(f"{r.model_condition:<6} {r.device:<6} {r.batch_size:<6} {r.mode:<14} "
              f"{r.encoder_latency_median:<12.2f} {r.adapter_latency_median:<12.2f} "
              f"{r.total_latency_median:<12.2f} {r.total_throughput_qps:<12.1f} {ndcg:<8}")

    print(f"\nResults saved to: {output_base}")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")
    if config.save_raw_latency:
        print(f"  - Raw latencies: {raw_path}")


if __name__ == "__main__":
    main()