"""Latency profiling utilities for benchmarking."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class LatencyStats:
    """Latency statistics for a benchmark run."""
    mean_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_qps: float
    raw_latencies_ms: List[float]


class LatencyProfiler:
    """Profiles latency of model operations."""

    def __init__(
        self,
        warmup_steps: int = 20,
        repetitions: int = 100,
        device: Optional[torch.device] = None,
    ):
        self.warmup_steps = warmup_steps
        self.repetitions = repetitions
        self.device = device or torch.device("cpu")
        self.is_cuda = self.device.type == "cuda"

    def _synchronize(self):
        """Synchronize CUDA if on GPU."""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    def profile(
        self,
        fn,
        *args,
        **kwargs,
    ) -> LatencyStats:
        """Profile a function and return latency statistics.

        Args:
            fn: Function to profile
            *args, **kwargs: Arguments to pass to the function

        Returns:
            LatencyStats with timing results
        """
        # Warmup
        for _ in range(self.warmup_steps):
            fn(*args, **kwargs)
        self._synchronize()

        # Timed runs
        latencies = []
        for _ in range(self.repetitions):
            self._synchronize()
            start = time.perf_counter()
            fn(*args, **kwargs)
            self._synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        return self._compute_stats(latencies)

    def _compute_stats(self, latencies: List[float]) -> LatencyStats:
        """Compute statistics from raw latencies."""
        import numpy as np

        latencies_arr = np.array(latencies)
        mean_ms = float(latencies_arr.mean())
        median_ms = float(np.median(latencies_arr))
        p95_ms = float(np.percentile(latencies_arr, 95))
        std_ms = float(latencies_arr.std(ddof=1))
        min_ms = float(latencies_arr.min())
        max_ms = float(latencies_arr.max())

        # Throughput: total queries / total time
        total_time_s = latencies_arr.sum() / 1000.0
        total_queries = len(latencies_arr)
        throughput_qps = total_queries / total_time_s if total_time_s > 0 else 0.0

        return LatencyStats(
            mean_ms=mean_ms,
            median_ms=median_ms,
            p95_ms=p95_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            throughput_qps=throughput_qps,
            raw_latencies_ms=latencies,
        )

    def profile_with_batch(
        self,
        fn,
        batch_size: int,
        *args,
        **kwargs,
    ) -> LatencyStats:
        """Profile a batched function and return per-query latency statistics.

        Args:
            fn: Function to profile (should process a batch)
            batch_size: Batch size for the function
            *args, **kwargs: Arguments to pass to the function

        Returns:
            LatencyStats with per-query timing results
        """
        # Warmup
        for _ in range(self.warmup_steps):
            fn(*args, **kwargs)
        self._synchronize()

        # Timed runs
        latencies = []
        for _ in range(self.repetitions):
            self._synchronize()
            start = time.perf_counter()
            fn(*args, **kwargs)
            self._synchronize()
            end = time.perf_counter()
            batch_latency_ms = (end - start) * 1000
            per_query_ms = batch_latency_ms / batch_size
            latencies.append(per_query_ms)

        return self._compute_stats(latencies)


@contextmanager
def timed_section(name: str, device: Optional[torch.device] = None):
    """Context manager for timing a code section."""
    is_cuda = device is not None and device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    yield
    if is_cuda:
        torch.cuda.synchronize(device)
    end = time.perf_counter()
    print(f"[{name}] {(end - start) * 1000:.2f} ms")


def measure_encoder_latency(model, texts: List[str], profiler: LatencyProfiler) -> LatencyStats:
    """Measure encoder-only latency."""
    return profiler.profile(model.encode, texts)


def measure_adapter_latency(model, embeddings: torch.Tensor, profiler: LatencyProfiler) -> LatencyStats:
    """Measure adapter-only latency."""
    return profiler.profile(model.adapt, embeddings)


def measure_encode_and_adapt_latency(model, texts: List[str], profiler: LatencyProfiler) -> LatencyStats:
    """Measure encode + adapter latency."""
    return profiler.profile(model.encode_and_adapt, texts)