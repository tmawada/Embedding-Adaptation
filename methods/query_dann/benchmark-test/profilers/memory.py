"""Memory profiling utilities for benchmarking."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import psutil
import torch


@dataclass
class MemoryStats:
    """Memory statistics for a benchmark run."""
    # Process memory
    process_rss_mb: float
    process_vms_mb: float
    peak_rss_mb: Optional[float] = None

    # GPU memory (if available)
    gpu_allocated_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
    gpu_peak_allocated_mb: Optional[float] = None
    gpu_peak_reserved_mb: Optional[float] = None

    # System memory
    system_total_mb: Optional[float] = None
    system_available_mb: Optional[float] = None
    system_used_mb: Optional[float] = None
    swap_total_mb: Optional[float] = None
    swap_used_mb: Optional[float] = None


class MemoryProfiler:
    """Profiles memory usage during benchmark runs."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.is_cuda = self.device.type == "cuda"
        self.process = psutil.Process(os.getpid())

        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # Process memory
        mem_info = self.process.memory_info()
        process_rss_mb = mem_info.rss / (1024 ** 2)
        process_vms_mb = mem_info.vms / (1024 ** 2)

        # Try to get peak RSS (Linux-specific)
        peak_rss_mb = None
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        peak_rss_mb = int(line.split()[1]) / 1024.0  # kB to MB
                        break
        except (FileNotFoundError, PermissionError, IndexError):
            pass

        # System memory
        sys_mem = psutil.virtual_memory()
        system_total_mb = sys_mem.total / (1024 ** 2)
        system_available_mb = sys_mem.available / (1024 ** 2)
        system_used_mb = sys_mem.used / (1024 ** 2)

        # Swap memory
        swap_mem = psutil.swap_memory()
        swap_total_mb = swap_mem.total / (1024 ** 2)
        swap_used_mb = swap_mem.used / (1024 ** 2)

        # GPU memory
        gpu_allocated_mb = None
        gpu_reserved_mb = None
        gpu_peak_allocated_mb = None
        gpu_peak_reserved_mb = None

        if self.is_cuda:
            gpu_allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            gpu_reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            gpu_peak_allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            gpu_peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)

        return MemoryStats(
            process_rss_mb=process_rss_mb,
            process_vms_mb=process_vms_mb,
            peak_rss_mb=peak_rss_mb,
            gpu_allocated_mb=gpu_allocated_mb,
            gpu_reserved_mb=gpu_reserved_mb,
            gpu_peak_allocated_mb=gpu_peak_allocated_mb,
            gpu_peak_reserved_mb=gpu_peak_reserved_mb,
            system_total_mb=system_total_mb,
            system_available_mb=system_available_mb,
            system_used_mb=system_used_mb,
            swap_total_mb=swap_total_mb,
            swap_used_mb=swap_used_mb,
        )

    def sample_memory(self, interval: float = 0.1, duration: float = 10.0) -> List[MemoryStats]:
        """Sample memory over a duration at given interval.

        Args:
            interval: Sampling interval in seconds
            duration: Total duration in seconds

        Returns:
            List of MemoryStats samples
        """
        import time

        samples = []
        end_time = time.time() + duration
        while time.time() < end_time:
            samples.append(self.get_memory_stats())
            time.sleep(interval)
        return samples


def get_gpu_utilization(device: Optional[torch.device] = None) -> Optional[float]:
    """Get GPU utilization percentage using nvidia-smi.

    Returns:
        GPU utilization percentage or None if not available
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def log_gpu_memory_summary(device: torch.device) -> str:
    """Get a formatted GPU memory summary."""
    if device.type != "cuda":
        return "Not on CUDA device"

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    return (
        f"GPU Memory Summary:\n"
        f"  Allocated: {allocated:.2f} GB\n"
        f"  Reserved: {reserved:.2f} GB\n"
        f"  Peak Allocated: {peak_allocated:.2f} GB\n"
        f"  Peak Reserved: {peak_reserved:.2f} GB"
    )