"""Compute resource profiler for local LLM inference.

Tracks latency, GPU memory, system memory, throughput, and optional GPU power.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import psutil
import torch


@dataclass
class ComputeMetrics:
    """Compute resource metrics for a single inference run."""
    # Latency (seconds)
    latency_s: float = 0.0
    latency_ms: float = 0.0
    
    # GPU Memory (MB)
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    
    # System Memory (MB)
    process_rss_mb: float = 0.0
    process_vms_mb: float = 0.0
    system_available_mb: float = 0.0
    
    # Throughput
    tokens_per_second: float = 0.0
    queries_per_second: float = 0.0
    
    # GPU Power (Watts) - optional
    gpu_power_avg_w: float = 0.0
    gpu_power_max_w: float = 0.0
    gpu_temperature_avg_c: float = 0.0
    gpu_temperature_max_c: float = 0.0
    gpu_utilization_avg_pct: float = 0.0
    gpu_utilization_max_pct: float = 0.0
    
    # Model info
    model_name: str = ""
    batch_size: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ComputeProfiler:
    """Profile compute resources during inference."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        track_gpu_memory: bool = True,
        track_system_memory: bool = True,
        track_power: bool = False,
        sample_interval_ms: int = 10,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.track_gpu_memory = track_gpu_memory and self.device.type == "cuda"
        self.track_system_memory = track_system_memory
        self.track_power = track_power and self._nvidia_smi_available()
        self.sample_interval = sample_interval_ms / 1000.0
        
        self._sampling = False
        self._samples: List[Dict] = []
        self._sampling_thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid())
    
    def _nvidia_smi_available(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            subprocess.run(["nvidia-smi", "--version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
    
    def _sample_gpu_memory(self) -> Dict[str, float]:
        """Sample GPU memory usage."""
        if not self.track_gpu_memory:
            return {}
        return {
            "gpu_allocated_mb": torch.cuda.memory_allocated(self.device) / 1e6,
            "gpu_reserved_mb": torch.cuda.memory_reserved(self.device) / 1e6,
            "gpu_peak_mb": torch.cuda.max_memory_allocated(self.device) / 1e6,
        }
    
    def _sample_system_memory(self) -> Dict[str, float]:
        """Sample system memory usage."""
        if not self.track_system_memory:
            return {}
        mem = self._process.memory_info()
        sys_mem = psutil.virtual_memory()
        return {
            "process_rss_mb": mem.rss / 1e6,
            "process_vms_mb": mem.vms / 1e6,
            "system_available_mb": sys_mem.available / 1e6,
        }
    
    def _sample_gpu_power(self) -> Dict[str, float]:
        """Sample GPU power/temperature/utilization via nvidia-smi."""
        if not self.track_power:
            return {}
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                if len(values) >= 3:
                    return {
                        "gpu_power_w": float(values[0]),
                        "gpu_temperature_c": float(values[1]),
                        "gpu_utilization_pct": float(values[2]),
                    }
        except Exception:
            pass
        return {}
    
    def _sampling_loop(self):
        """Background sampling loop."""
        while self._sampling:
            sample = {}
            sample.update(self._sample_gpu_memory())
            sample.update(self._sample_system_memory())
            sample.update(self._sample_gpu_power())
            sample["timestamp"] = time.monotonic()
            self._samples.append(sample)
            time.sleep(self.sample_interval)
    
    def start_sampling(self):
        """Start background resource sampling."""
        if self._sampling:
            return
        self._sampling = True
        self._samples = []
        # Reset peak memory tracking
        if self.track_gpu_memory:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sampling_thread.start()
    
    def stop_sampling(self) -> Dict[str, float]:
        """Stop sampling and return aggregated metrics."""
        if not self._sampling:
            return {}
        self._sampling = False
        if self._sampling_thread:
            self._sampling_thread.join(timeout=1.0)
        
        if not self._samples:
            return {}
        
        # Aggregate samples
        gpu_alloc = [s.get("gpu_allocated_mb", 0) for s in self._samples]
        gpu_reserved = [s.get("gpu_reserved_mb", 0) for s in self._samples]
        rss = [s.get("process_rss_mb", 0) for s in self._samples]
        power = [s.get("gpu_power_w", 0) for s in self._samples]
        temp = [s.get("gpu_temperature_c", 0) for s in self._samples]
        util = [s.get("gpu_utilization_pct", 0) for s in self._samples]
        
        return {
            "gpu_memory_allocated_mb": max(gpu_alloc) if gpu_alloc else 0,
            "gpu_memory_reserved_mb": max(gpu_reserved) if gpu_reserved else 0,
            "gpu_memory_peak_mb": torch.cuda.max_memory_allocated(self.device) / 1e6 if self.track_gpu_memory else 0,
            "process_rss_mb": max(rss) if rss else 0,
            "process_vms_mb": self._process.memory_info().vms / 1e6,
            "system_available_mb": psutil.virtual_memory().available / 1e6,
            "gpu_power_avg_w": sum(power) / len(power) if power else 0,
            "gpu_power_max_w": max(power) if power else 0,
            "gpu_temperature_avg_c": sum(temp) / len(temp) if temp else 0,
            "gpu_temperature_max_c": max(temp) if temp else 0,
            "gpu_utilization_avg_pct": sum(util) / len(util) if util else 0,
            "gpu_utilization_max_pct": max(util) if util else 0,
        }
    
    @contextmanager
    def profile(self, model_name: str = "", batch_size: int = 1):
        """Context manager for profiling a single inference run."""
        # Reset peak memory
        if self.track_gpu_memory:
            torch.cuda.reset_peak_memory_stats(self.device)
        
        # Start sampling
        self.start_sampling()
        
        # Start timing
        t0 = time.monotonic()
        
        try:
            yield
        finally:
            # Stop timing
            latency_s = time.monotonic() - t0
            
            # Stop sampling
            metrics = self.stop_sampling()
            
            # Create metrics object
            result = ComputeMetrics(
                latency_s=latency_s,
                latency_ms=latency_s * 1000,
                model_name=model_name,
                batch_size=batch_size,
                **metrics,
            )
            
            # Store result
            self.last_result = result
    
    def get_last_result(self) -> Optional[ComputeMetrics]:
        """Get the last profiling result."""
        return getattr(self, 'last_result', None)
    
    def to_dict(self) -> Dict:
        """Convert last result to dictionary."""
        result = self.get_last_result()
        return asdict(result) if result else {}


class BatchProfiler:
    """Profile a batch of inferences and compute aggregate statistics."""
    
    def __init__(self, profiler: ComputeProfiler):
        self.profiler = profiler
        self.results: List[ComputeMetrics] = []
    
    def profile_batch(
        self,
        fn,
        inputs: List,
        model_name: str = "",
        batch_size: int = 1,
        warmup: int = 0,
    ) -> List[ComputeMetrics]:
        """Profile a function over a batch of inputs."""
        # Warmup runs
        for _ in range(warmup):
            fn(inputs[0] if inputs else None)
        
        # Synchronize GPU before profiling
        if self.profiler.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Profile each input
        for i, inp in enumerate(inputs):
            with self.profiler.profile(model_name=model_name, batch_size=batch_size):
                fn(inp)
            
            # Synchronize after each for accurate timing
            if self.profiler.device.type == "cuda":
                torch.cuda.synchronize()
            
            result = self.profiler.get_last_result()
            if result:
                self.results.append(result)
        
        return self.results
    
    def get_summary(self) -> Dict:
        """Get aggregate statistics."""
        if not self.results:
            return {}
        
        latencies = [r.latency_ms for r in self.results]
        gpu_mem = [r.gpu_memory_peak_mb for r in self.results]
        rss = [r.process_rss_mb for r in self.results]
        tokens = [r.total_tokens for r in self.results if r.total_tokens > 0]
        
        total_time = sum(r.latency_s for r in self.results)
        total_tokens = sum(tokens)
        
        return {
            "num_runs": len(self.results),
            "latency_ms": {
                "mean": sum(latencies) / len(latencies),
                "median": sorted(latencies)[len(latencies) // 2],
                "p95": sorted(latencies)[int(0.95 * len(latencies))],
                "p99": sorted(latencies)[int(0.99 * len(latencies))],
                "min": min(latencies),
                "max": max(latencies),
                "std": (sum((x - sum(latencies)/len(latencies))**2 for x in latencies) / len(latencies))**0.5,
            },
            "gpu_memory_peak_mb": {
                "mean": sum(gpu_mem) / len(gpu_mem) if gpu_mem else 0,
                "max": max(gpu_mem) if gpu_mem else 0,
            },
            "process_rss_mb": {
                "mean": sum(rss) / len(rss) if rss else 0,
                "max": max(rss) if rss else 0,
            },
            "throughput": {
                "queries_per_second": len(self.results) / total_time if total_time > 0 else 0,
                "tokens_per_second": total_tokens / total_time if total_time > 0 and total_tokens > 0 else 0,
            },
            "total_time_s": total_time,
            "total_tokens": total_tokens,
        }
    
    def save_json(self, path: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "summary": self.get_summary(),
            "runs": [asdict(r) for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()
        if not summary:
            print("No results to summarize")
            return
        
        print(f"\n{'='*60}")
        print(f"COMPUTE PROFILE SUMMARY ({summary['num_runs']} runs)")
        print(f"{'='*60}")
        
        lat = summary["latency_ms"]
        print(f"Latency (ms):")
        print(f"  Mean: {lat['mean']:.2f}  Median: {lat['median']:.2f}  "
              f"P95: {lat['p95']:.2f}  P99: {lat['p99']:.2f}")
        print(f"  Min: {lat['min']:.2f}  Max: {lat['max']:.2f}  Std: {lat['std']:.2f}")
        
        gpu = summary["gpu_memory_peak_mb"]
        if gpu["max"] > 0:
            print(f"GPU Memory (MB): Mean: {gpu['mean']:.1f}  Peak: {gpu['max']:.1f}")
        
        rss = summary["process_rss_mb"]
        print(f"Process RSS (MB): Mean: {rss['mean']:.1f}  Peak: {rss['max']:.1f}")
        
        tp = summary["throughput"]
        print(f"Throughput: {tp['queries_per_second']:.2f} q/s")
        if tp["tokens_per_second"] > 0:
            print(f"  Tokens/s: {tp['tokens_per_second']:.1f}")
        
        print(f"Total time: {summary['total_time_s']:.2f}s")


def profile_model_inference(
    model_name: str,
    model_fn,
    inputs: List,
    device: Optional[torch.device] = None,
    batch_size: int = 1,
    warmup: int = 3,
    track_power: bool = False,
    output_path: Optional[str] = None,
) -> Dict:
    """Convenience function to profile model inference."""
    profiler = ComputeProfiler(
        device=device,
        track_gpu_memory=True,
        track_system_memory=True,
        track_power=track_power,
    )
    batch_profiler = BatchProfiler(profiler)
    
    results = batch_profiler.profile_batch(
        model_fn,
        inputs,
        model_name=model_name,
        batch_size=batch_size,
        warmup=warmup,
    )
    
    summary = batch_profiler.get_summary()
    batch_profiler.print_summary()
    
    if output_path:
        batch_profiler.save_json(output_path)
    
    return summary


if __name__ == "__main__":
    # Simple test
    profiler = ComputeProfiler(track_power=True)
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"nvidia-smi available: {profiler._nvidia_smi_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")