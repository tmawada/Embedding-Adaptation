# Local LLM Integration Summary

## Overview
Successfully integrated Qwen2.5 3B (q4_K_M, 1.9 GB) via Ollama for query normalization, enabling fair comparison with API-based LLMs and Query-DANN adapter.

## Components Created

### 1. `local_llm/client.py` - Ollama HTTP Client
- Synchronous and async client for Ollama API
- Latency tracking per request
- Token counting (prompt/completion/total)
- Model management (list, pull, show)
- Error handling with retries

### 2. `local_llm/normalize.py` - Query Normalization
- Uses **exact same prompt template** as Gemini baseline for fair comparison
- Few-shot Indonesian informal → formal query conversion
- Resumable JSONL caching
- Batch processing support
- Per-query latency and token tracking

### 3. `local_llm/evaluate.py` - Evaluation Script
- Reuses existing `methods/eval.py` infrastructure
- Computes nDCG@10, MRR@10, Recall@100
- Paired bootstrap significance testing
- Compares against:
  - Base BGE-M3 on formal query (upper bound)
  - Base BGE-M3 on informal query (zero-shot)
  - Local LLM-normalized queries
  - Query-DANN adapter (optional)

### 4. `local_llm/compute_profiler.py` - Compute Resource Profiler
- Latency tracking (mean, median, p95, p99)
- GPU memory tracking (via torch.cuda)
- System memory tracking (via psutil)
- GPU power/temperature/utilization (via nvidia-smi)
- Throughput calculation (queries/s, tokens/s)
- JSON export for reporting

### 5. `local_llm/configs/qwen2.5_3b.yaml` - Configuration
- Model parameters (temperature, top_p, max_tokens, etc.)
- Normalization settings (shots, batch_size, concurrency)
- Profiling settings
- Output settings

## Results Summary

### Test Split (553 queries, 500K passages)

| Condition | nDCG@10 | MRR@10 | Recall@100 | Gap Closed |
|-----------|---------|--------|------------|------------|
| **1. Base BGE-M3 \| formal (upper bound)** | **0.6141** | **0.7196** | **0.9375** | — |
| **2. Base BGE-M3 \| informal (zero-shot)** | 0.5362 | 0.6364 | 0.8986 | 0.0% |
| **3. Base BGE-M3 \| Qwen2.5 3B (local)** | **0.5850** | **0.6825** | **0.9124** | **62.7%** |
| **4. Query-DANN adapter \| informal** | 0.5488 | 0.6554 | 0.8966 | 16.2% |

### Significance Tests (paired bootstrap, 1000 samples)

| Comparison | nDCG@10 | MRR@10 | Recall@100 |
|------------|---------|--------|------------|
| Local LLM vs Zero-shot | **+0.0489*** | **+0.0461*** | +0.0137 |
| Local LLM vs Formal | -0.0290* | -0.0370* | -0.0251* |
| Local LLM vs Query-DANN | **+0.0363*** | **+0.0272*** | +0.0157 |

* = p < 0.05

### Dev Split (372 queries)

| Condition | nDCG@10 | Gap Closed |
|-----------|---------|------------|
| Formal (upper bound) | 0.6181 | — |
| Zero-shot informal | 0.5390 | 0.0% |
| **Qwen2.5 3B (local)** | **0.5772** | **48.3%** |
| Query-DANN adapter | 0.5614 | 28.3% |

On dev split, the difference between Local LLM and Query-DANN is **not statistically significant**.

### Compute Resources (RTX 3090, q4_K_M)

| Metric | Value |
|--------|-------|
| **Latency (mean)** | 57.9 ms/query |
| **Latency (median)** | 56.5 ms/query |
| **Latency (p95)** | 73.6 ms/query |
| **Throughput** | 17.3 queries/second |
| **GPU Power (avg)** | ~150 W |
| **GPU Power (peak)** | ~280 W |
| **GPU Utilization** | 50-70% |
| **GPU Temperature** | 55-57°C |
| **Process RSS** | 579 MB |
| **Tokens per query** | ~274 |

### Comparison with API LLM (Gemini 3.5 Flash-Lite)

| Metric | Gemini 3.5 Flash-Lite | Qwen2.5 3B (local) |
|--------|----------------------|-------------------|
| **Gap Closure (test)** | 79.7% | 62.7% |
| **Gap Closure (dev)** | N/A | 48.3% |
| **Latency/query** | ~50 ms (batched) | ~58 ms |
| **Infrastructure** | Cloud API | Local (same GPU) |
| **Cost/query** | API pricing | Free (local) |
| **Privacy** | External | Local |

**Key Insight**: Local Qwen2.5 3B achieves **78% of the API LLM's gap closure** (62.7% vs 79.7%) while running on the same hardware as other models, enabling truly fair comparison.

## Usage

### Normalize Queries
```bash
# Full test split
python3 -m local_llm.normalize --split test --shots 3

# Dev split
python3 -m local_llm.normalize --split dev --shots 3

# Quick test (5 queries)
python3 -m local_llm.normalize --split test --limit 5
```

### Evaluate Normalized Queries
```bash
# Basic evaluation
python3 -m local_llm.evaluate --rewrites local_llm/cache/qwen2.5_3b-instruct-q4_K_M_test.jsonl --split test

# With Query-DANN comparison
python3 -m local_llm.evaluate --rewrites local_llm/cache/qwen2.5_3b-instruct-q4_K_M_test.jsonl --split test \
    --run query_dann=methods/query_dann/outputs/harmonized/best
```

### Profile Compute Resources
```bash
python3 -m local_llm.profile_compute
```

## Files Generated

- `local_llm/cache/qwen2.5_3b-instruct-q4_K_M_test.jsonl` - 553 normalized test queries
- `local_llm/cache/qwen2.5_3b-instruct-q4_K_M_dev.jsonl` - 372 normalized dev queries
- `local_llm/cache/compute_profile.json` - Detailed compute profiling results

## Next Steps

1. **Add to benchmark suite**: Integrate as M3 condition in `methods/query_dann/benchmark-test/`
2. **Multi-model comparison**: Test other local models (Llama 3.1 8B, Gemma 2 9B)
3. **Quantization study**: Compare q4_K_M vs q8_0 vs fp16 for quality/compute tradeoff
4. **Batch optimization**: Test larger batch sizes for throughput optimization
5. **Energy estimation**: Add total energy per query (power × time)