# BGE-M3 vs Query-DANN Benchmark

This benchmark suite implements the specification in `query_dann_benchmark_spec.md` to compare:

- **M0 — BGE-M3 baseline**: Original `BAAI/bge-m3`, no adapter
- **M1 — Untrained adapter control**: Same frozen encoder + newly initialized, untrained adapter
- **M2 — Query-DANN**: Same frozen encoder + trained adapter from checkpoint

Evaluated on **CPU and GPU** in two modes:
1. **Model-only**: Text encoding and adapter transformation (no vector search)
2. **Full pipeline**: Text encoding, adapter transformation, and retrieval against precomputed document index

## Quick Start

```bash
# From the benchmark-test directory
cd /home/yd/adapt-dann-retrieve/bge_dann/methods/query_dann/benchmark-test

# Quick smoke test (reduced repetitions)
python run_benchmark.py --quick --model baseline --device cpu --modes model_only

# Full benchmark on CPU (all models)
python run_benchmark.py --model all --device cpu --modes model_only full_pipeline \
  --checkpoint ../outputs/query_dann/best

# Full benchmark on GPU (if available)
python run_benchmark.py --model all --device cuda --modes model_only full_pipeline \
  --checkpoint ../outputs/query_dann/best
```

## Configuration

The benchmark is configured via `configs/benchmark.yaml`:

```yaml
experiment:
  name: query_dann_efficiency
  seed: 42

models:
  encoder: BAAI/bge-m3
  baseline: true
  adapter_untrained: true
  query_dann: true

benchmark:
  modes: [model_only, full_pipeline]
  devices: [cpu, cuda]
  batch_sizes: [1, 8, 32, 64]
  warmup_steps: 20
  repetitions: 100
  top_k: 100

runtime:
  precision: fp32
  max_length: 512
  normalize_embeddings: true

output:
  directory: reports
  save_raw_latency: true
  save_memory_samples: true
  save_hardware_metadata: true
```

## CLI Reference

```bash
python run_benchmark.py [OPTIONS]

Options:
  --config PATH           Path to benchmark config YAML (default: configs/benchmark.yaml)
  --model {baseline,query_dann,all}  Model condition (default: all)
  --checkpoint PATH       Query-DANN checkpoint directory for M2
  --device {cpu,cuda,all} Device to run on (default: all)
  --modes MODE [MODE ...] Benchmark modes: model_only, full_pipeline
  --batch-sizes N [N ...] Batch sizes to test
  --output-dir PATH       Output directory (overrides config)
  --run-id ID             Custom run ID
  --quick                 Quick test with reduced repetitions
  --seed INT              Random seed (default: 42)
```

## Output Structure

Results are saved to `reports/<run_id>/`:

```
reports/<run_id>/
├── hardware/
│   └── system_info.json          # Hardware/software metadata
├── parameters/
│   ├── M0_params.json            # Parameter counts for M0
│   ├── M1_params.json            # Parameter counts for M1
│   └── M2_params.json            # Parameter counts for M2
├── model_only/
│   ├── cpu.csv                   # Model-only results (CPU)
│   └── cuda.csv                  # Model-only results (GPU)
├── full_pipeline/
│   ├── cpu.csv                   # Full-pipeline results (CPU)
│   └── cuda.csv                  # Full-pipeline results (GPU)
├── raw/
│   └── latency_observations.json # Raw latency measurements
└── summary/
    ├── benchmark_results.csv     # Combined results (CSV)
    └── benchmark_results.json    # Combined results (JSON)
```

## Results Schema

Each result row contains:

| Field | Description |
|-------|-------------|
| `run_id` | Unique run identifier |
| `timestamp` | ISO timestamp |
| `model_condition` | M0, M1, or M2 |
| `device` | cpu or cuda |
| `precision` | fp32, fp16, bf16 |
| `batch_size` | Batch size |
| `max_length` | Max sequence length |
| `mode` | model_only or full_pipeline |
| `encoder_latency_median` | Median encoder latency (ms/query) |
| `encoder_latency_p95` | P95 encoder latency (ms/query) |
| `encoder_throughput_qps` | Encoder throughput (queries/sec) |
| `adapter_latency_median` | Median adapter latency (ms/query) |
| `adapter_latency_p95` | P95 adapter latency (ms/query) |
| `adapter_throughput_qps` | Adapter throughput (queries/sec) |
| `total_latency_median` | Median total latency (ms/query) |
| `total_latency_p95` | P95 total latency (ms/query) |
| `total_throughput_qps` | End-to-end throughput (queries/sec) |
| `ndcg_at_10` | nDCG@10 (full_pipeline only) |
| `mrr_at_10` | MRR@10 (full_pipeline only) |
| `recall_at_100` | Recall@100 (full_pipeline only) |
| `peak_rss_mb` | Peak process RSS (MB) |
| `peak_vram_mb` | Peak GPU VRAM (MB) |
| `raw_latencies` | JSON array of per-query latencies (ms) |

## Paper-Ready Tables

The benchmark generates data for these tables:

### Table 1: Model Complexity
| Model | Encoder params | Adapter params | Discriminator params (train only) | Inference params | Checkpoint size |
|-------|----------------|----------------|-----------------------------------|------------------|-----------------|
| M0    |                | —              | —                                 |                  |                 |
| M1    |                |                | —                                 |                  |                 |
| M2    |                |                |                                   |                  |                 |

### Table 2: Model-Only Efficiency
| Model | Device | Batch | Median latency | P95 latency | Throughput | Peak RAM | Peak VRAM |
|-------|--------|-------|----------------|-------------|------------|----------|-----------|

### Table 3: Full-Pipeline Efficiency
| Model | Device | Batch | Encoder time | Adapter time | Retrieval time | End-to-end | Throughput |
|-------|--------|-------|--------------|--------------|----------------|------------|------------|

### Table 4: Retrieval Effectiveness
| Model | Query type | nDCG@10 | MRR@10 | Recall@100 |
|-------|------------|---------|--------|------------|

### Table 5: Training Cost
| Stage | Wall time | Peak RAM | Peak VRAM | CPU util | GPU util |
|-------|-----------|----------|-----------|----------|----------|

## Requirements

- PyTorch with CUDA (for GPU benchmarks)
- transformers, accelerate
- psutil
- pyyaml
- numpy

Install with:
```bash
pip install torch transformers accelerate psutil pyyaml numpy
```

## Notes

- **FP32** is used for primary comparison as per spec
- **Warmup steps** and **CUDA synchronization** are applied for accurate timing
- **Raw observations** are saved for statistical analysis
- **Discriminator** is excluded from inference parameter counts and latency
- **Same corpus embeddings/index** used across all conditions
- **OOM configurations** are recorded and skipped

## Validation

Before running full experiments, run the smoke test:

```bash
python run_benchmark.py --quick --model all --device cpu --modes model_only full_pipeline \
  --checkpoint ../outputs/query_dann/best
```

Verify:
1. M0 reproduces existing baseline outputs
2. M2 reproduces existing Query-DANN inference outputs
3. No OOM errors on target batch sizes