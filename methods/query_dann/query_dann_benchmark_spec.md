# Implementation Specification: BGE-M3 vs Query-DANN Benchmark

## 1. Objective

Implement a reproducible benchmark in the existing research repository
comparing:

-   **M0 --- BGE-M3 baseline:** original `BAAI/bge-m3`, no adapter.
-   **M1 --- BGE-M3 + untrained adapter:** same frozen encoder followed
    by a newly initialized, untrained adapter. This is an optional
    control for adapter execution overhead.
-   **M2 --- BGE-M3 + Query-DANN:** same frozen encoder followed by the
    trained adapter loaded from the existing Query-DANN checkpoint.

Evaluate on **CPU and GPU**, in two modes:

1.  **Model-only:** text encoding and adapter transformation, without
    vector search.
2.  **Full pipeline:** text encoding, adapter transformation if
    applicable, and retrieval against the same precomputed document
    index.

Also report training/preparation cost separately. The goal is to
quantify retrieval effectiveness and the computational overhead of
embedding-level adversarial query adaptation.

## 2. Inspect the repository before coding

Do not assume class names, file paths, checkpoint formats, or tensor
shapes. First inspect the actual source code and identify:

-   BGE-M3 loading, tokenization, pooling, dense embedding extraction,
    and normalization.
-   Residual adapter implementation and initialization.
-   Query-DANN wrapper.
-   Gradient Reversal Layer (GRL).
-   Domain discriminator.
-   EmbeddingStore and cache-generation logic.
-   Training loop, optimizer, and checkpoint save/load logic.
-   Existing inference/search and evaluation/retrieval implementations.

Read the relevant files fully. Record actual tensor shapes, precision,
max sequence length, normalization behavior, and checkpoint contents.

The existing project summary describes BGE-M3 as a frozen
1024-dimensional dense encoder; a residual bottleneck adapter (typically
1024 → 128 → 1024); a training-only discriminator; and the objective:

`L_total = L_InfoNCE + gamma * L_domain + mu * L_preserve`

It also describes cached embeddings and exact inner-product search over
L2-normalized corpus embeddings. **Verify all these details against the
code.** If the implementation differs, preserve actual behavior and
document the discrepancy.

Do not rewrite Query-DANN from scratch or create a merely similar
architecture. Reuse the real project classes and checkpoint loader.

## 3. Experimental conditions

  ---------------------------------------------------------------------------
  ID             Condition      Encoder        Adapter        Discriminator
                                                              at inference
  -------------- -------------- -------------- -------------- ---------------
  M0             BGE-M3         Original       None           No
                 baseline       frozen BGE-M3                 

  M1             Untrained      Same frozen    Newly          No
                 adapter        BGE-M3         initialized,   
                 control                       untrained      

  M2             Query-DANN     Same frozen    Trained        No
                                BGE-M3         adapter from   
                                               checkpoint     
  ---------------------------------------------------------------------------

M0 vs M2 is the primary comparison. M1 is a recommended control to
separate adapter execution overhead from the effect of training.

Do not count the discriminator as an inference component. Report its
parameter count separately as training-only. Verify that the untrained
adapter initialization is identity-like; do not assume exact identity
without checking residual, normalization, and initialization behavior.

## 4. Benchmark modes

### 4.1 Model-only

Measure: - BGE-M3 text-encoding latency. - Adapter latency, where
applicable. - Total model-only latency. - Throughput and resource use.

Do not include vector search, index loading, or retrieval metric
calculation.

Expected flow:

``` text
M0: text → BGE-M3 → embedding
M1: text → BGE-M3 → untrained adapter → embedding
M2: text → BGE-M3 → trained adapter → adapted embedding
```

### 4.2 Full pipeline

Measure: - Text encoding. - Adapter transformation, where applicable. -
Vector retrieval. - End-to-end latency.

Report encoder, adapter, retrieval, and total latency separately.

All conditions must use the same: - Corpus embeddings and index. -
Retrieval backend and implementation. - Similarity function. - Top-K. -
Query split and query ordering.

Use the existing retrieval implementation where possible. Do not use
FAISS for one condition and brute-force PyTorch for another. Verify
whether the current implementation uses L2-normalized vectors and exact
inner product; preserve the same behavior for every condition.

## 5. CPU and GPU

Run CPU and CUDA experiments separately. If CUDA is unavailable, mark
GPU experiments as skipped with a reason.

Record: - CPU model and core/thread counts. - System RAM capacity. - GPU
model and VRAM. - Driver and CUDA versions. - OS, Python, PyTorch,
Transformers, and embedding library versions. - PyTorch CPU thread
count. - Device and precision for each run.

Control and record CPU thread settings. For GPU timing, call
`torch.cuda.synchronize()` before and after timed regions. Do not
compare CPU and GPU results as though only the model changed.

## 6. Controlled configuration

Create a shared config such as `benchmark/configs/benchmark.yaml`.

Suggested starting configuration:

``` yaml
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

Validate batch sizes and sequence length against available memory.
Record OOM configurations and continue with remaining runs. Use FP32 for
the primary comparison. Optional FP16/BF16 results must be separate
experiments.

Keep tokenizer, preprocessing, max length, precision, evaluation mode,
normalization, batch sizes, query inputs, retrieval code, and corpus
index identical across model conditions. Do not train or tune on test
data.

## 7. Latency and throughput

Measure batch latency and derive per-query latency. Report: - Mean. -
Median (p50). - p95. - Standard deviation. - Throughput
(queries/second).

Warm up before timing. Exclude model loading, tokenizer initialization,
first-time CUDA initialization, and index construction from steady-state
inference latency. Report cold-start costs separately if useful.

For CUDA, synchronize before and after timing. Save raw observations.
Calculate throughput as total queries processed divided by total
measured wall-clock time, not simply the inverse of median single-query
latency.

Measure encoder, adapter, retrieval, and total pipeline components
without double-counting work.

## 8. Parameter and checkpoint analysis

Count actual instantiated parameters and report:

1.  BGE-M3 total and frozen parameters.
2.  Adapter total and trainable parameters.
3.  Discriminator parameters (training only).
4.  Total inference parameters: encoder only for M0; encoder + adapter
    for M1/M2.
5.  Total training parameters for Query-DANN, distinguishing frozen
    encoder from trainable adapter and discriminator.
6.  Checkpoint file sizes for encoder, adapter, discriminator, and
    combined training checkpoint where measurable.

Calculate:

`adapter_parameter_ratio_percent = adapter_parameters / encoder_parameters * 100`

Do not estimate parameter counts from architecture descriptions when
actual model objects are available. Do not include discriminator
parameters in inference complexity.

## 9. Memory and utilization

### GPU

Measure: - Current allocated memory. - Current reserved memory. - Peak
allocated memory. - Peak reserved memory.

Reset peak statistics before each phase and synchronize before reading
them. Explain that PyTorch allocator metrics may not include every CUDA
allocation outside the PyTorch allocator.

### Process and system RAM

Use `psutil` to measure: - Process RSS. - Process VMS (label as virtual
memory, not physical RAM usage). - Peak RSS if supported. - System RAM
used/available. - Swap used/available if available.

### CPU/GPU utilization

Sample utilization during workload execution, not only after it. Record
sampling interval and collection method. Use an available supported GPU
monitoring interface (for example, `nvidia-smi` when present);
monitoring failure should not abort the benchmark.

Clarify that utilization depends on hardware, drivers, batch size,
sequence length, background load, and runtime settings. It is not an
intrinsic model property.

## 10. Retrieval effectiveness

Evaluate formal and informal queries separately:

-   M0 formal and informal.
-   M1 formal and informal if included in retrieval evaluation.
-   M2 informal (main proposed condition).
-   M2 formal (robustness check).

Metrics: - nDCG@10. - MRR@10. - Recall@100.

Use identical test query IDs, qrels, corpus, index, relevance handling,
retrieval code, and top-K. Reuse the existing evaluator where possible.
Preserve per-query metrics and use paired bootstrap on the same queries
if supported by the existing evaluation pipeline. Report mean, standard
deviation, confidence intervals, and p-values where appropriate.

Do not fabricate or substitute retrieval results.

## 11. Training and preparation cost

The existing implementation reportedly trains the adapter using cached
BGE-M3 embeddings. Verify this, then measure three separate stages.

### Stage 1 --- Embedding preparation

Measure generation of corpus, formal-query, informal-query, and required
hard-negative embeddings. Report wall-clock time, peak RAM/VRAM,
utilization, number of encoded items, batch size, and max sequence
length.

### Stage 2 --- Adapter/discriminator training

Measure training time, steps/second or samples/second, peak RAM/VRAM,
utilization, trainable parameters, epochs, optimizer, and training
configuration. Do not include BGE-M3 forward-pass time here if cached
embeddings are actually used.

### Stage 3 --- Dev evaluation and checkpoint selection

Measure dev evaluation time and checkpoint save time where separable;
record checkpoint size.

Calculate and report:

`total_adaptation_preparation_time = embedding_preparation_time + adapter_training_time + dev_evaluation_and_checkpoint_time`

Report each stage individually as well. Do not compare baseline
inference time directly with Query-DANN training time.

## 12. Suggested project structure

Keep benchmark code separate from the existing research code:

``` text
benchmark/
├── configs/
│   └── benchmark.yaml
├── models/
│   ├── baseline.py
│   └── query_dann.py
├── profilers/
│   ├── latency.py
│   ├── memory.py
│   ├── parameters.py
│   └── system.py
├── evaluation/
│   └── retrieval.py
├── reports/
│   ├── hardware/
│   ├── parameters/
│   ├── model_only/
│   ├── full_pipeline/
│   ├── training/
│   └── summary/
├── run_benchmark.py
└── README.md
```

Adapt this to existing repository conventions. Avoid duplicating model
code unnecessarily.

## 13. Suggested CLI

Provide a unified runner. Exact arguments may be adapted to the existing
project:

``` bash
# Baseline on CPU
python benchmark/run_benchmark.py \
  --model baseline --device cpu \
  --modes model_only full_pipeline

# Query-DANN on CPU
python benchmark/run_benchmark.py \
  --model query_dann \
  --checkpoint methods/query_dann/outputs/RUN_NAME/best \
  --device cpu --modes model_only full_pipeline

# Baseline on GPU
python benchmark/run_benchmark.py \
  --model baseline --device cuda \
  --modes model_only full_pipeline

# Query-DANN on GPU
python benchmark/run_benchmark.py \
  --model query_dann \
  --checkpoint methods/query_dann/outputs/RUN_NAME/best \
  --device cuda --modes model_only full_pipeline

# All available conditions and devices
python benchmark/run_benchmark.py \
  --model all --device all \
  --modes model_only full_pipeline
```

Document exact commands and required paths in the README.

## 14. Output files

Generate machine-readable CSV and JSON:

``` text
reports/
├── hardware/system_info.json
├── parameters/parameter_count.csv
├── model_only/cpu.csv
├── model_only/cuda.csv
├── full_pipeline/cpu.csv
├── full_pipeline/cuda.csv
├── training/training_cost.csv
├── raw/latency_observations.csv
├── raw/resource_samples.csv
└── summary/
    ├── benchmark_summary.csv
    └── benchmark_summary.json
```

Include run ID, timestamp, model/checkpoint identifier, device,
precision, batch size, max sequence length, warmup/repetition counts,
corpus/index identifier, query split, and software/hardware metadata.
Never overwrite prior runs; use unique run directories.

## 15. Paper-ready result tables

Prepare:

### Table 1 --- Model complexity

  -----------------------------------------------------------------------------
  Model          Encoder      Adapter   Discriminator    Inference   Checkpoint
                  params       params          params       params         size
                                      (training only)              
  --------- ------------ ------------ --------------- ------------ ------------

  -----------------------------------------------------------------------------

### Table 2 --- Model-only efficiency

  -------------------------------------------------------------------------------
  Model   Device       Batch    Median       p95   Throughput  Peak RAM Peak VRAM
                               latency   latency                        
  ------- -------- --------- --------- --------- ------------ --------- ---------

  -------------------------------------------------------------------------------

### Table 3 --- Full-pipeline efficiency

  ------------------------------------------------------------------------------------
  Model   Device       Batch   Encoder   Adapter   Retrieval   End-to-end   Throughput
                                  time      time        time      latency 
  ------- -------- --------- --------- --------- ----------- ------------ ------------

  ------------------------------------------------------------------------------------

### Table 4 --- Retrieval effectiveness

  Model   Query type     nDCG@10   MRR@10   Recall@100
  ------- ------------ --------- -------- ------------

### Table 5 --- Training cost

  ----------------------------------------------------------------------------
  Stage        Wall time     Peak RAM    Peak VRAM           CPU           GPU
                                                     utilization   utilization
  --------- ------------ ------------ ------------ ------------- -------------

  ----------------------------------------------------------------------------

Also create: - Latency vs batch size plots, separately for CPU and
GPU. - Retrieval improvement vs inference latency overhead plot. -
Optional memory vs batch size plot.

Do not insert illustrative or fabricated values into final reports. Mark
unavailable measurements explicitly.

## 16. Validity requirements

The benchmark must ensure: - Same BGE-M3 checkpoint for all
conditions. - Same tokenizer and preprocessing. - Same queries and query
ordering. - Same corpus embeddings/index. - Same retrieval
implementation and top-K. - Same precision in primary comparisons. -
Same batch sizes and max sequence length. - Same device for
within-device comparisons. - No test-set training or hyperparameter
tuning. - Model loading excluded from steady-state latency. - Warm-up
and correct CUDA synchronization. - Raw observations saved. - CPU thread
count recorded. - Discriminator excluded from inference parameter count
and inference latency. - Embedding preparation time separated from
adapter training time. - OOM/unavailable metrics explicitly recorded.

Do not claim Query-DANN is computationally more efficient merely because
its adapter has few parameters. BGE-M3 still performs the encoder
forward pass in both M0 and M2. Measure and report actual overhead.

## 17. Implementation workflow

Work incrementally:

1.  Inspect source code and report actual model, checkpoint, data, and
    retrieval interfaces.
2.  Present a concise integration plan and identify missing information.
3.  Implement wrappers by reusing existing classes.
4.  Implement parameter counting.
5.  Implement model-only benchmark.
6.  Implement full-pipeline benchmark using existing retrieval code.
7.  Implement CPU/GPU memory and utilization monitoring.
8.  Instrument embedding preparation, training, and checkpoint selection
    costs.
9.  Implement CSV/JSON outputs and hardware metadata.
10. Add README.
11. Run a small smoke test with a few queries.
12. Verify M0 reproduces existing baseline outputs and M2 reproduces
    existing Query-DANN inference outputs.
13. Only then run expensive corpus-wide benchmarks.

Do not launch full experiments before smoke tests and parity checks
pass.

## 18. Final deliverables

Provide: 1. Benchmark source code. 2. README with setup and execution
instructions. 3. Benchmark configuration. 4. Hardware/software metadata
capture. 5. CSV/JSON result generation. 6. Smoke-test output. 7.
Explanation of model-only vs full-pipeline measurements. 8. Limitations
and metrics that could not be collected. 9. Confirmation that all
conditions use the same encoder, corpus index, query split, and
retrieval implementation.

**Critical instruction:** Do not fabricate source interfaces, checkpoint
contents, parameter counts, benchmark results, or hardware measurements.
Inspect the repository and integrate with the actual implementation.
