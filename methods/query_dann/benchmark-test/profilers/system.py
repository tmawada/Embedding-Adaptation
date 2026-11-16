"""System and hardware metadata collection for benchmarking."""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import psutil
import torch


@dataclass
class HardwareInfo:
    """Hardware information."""
    # CPU
    cpu_model: str
    cpu_physical_cores: int
    cpu_logical_cores: int

    # Memory
    system_ram_total_gb: float
    system_ram_available_gb: float

    # CPU (optional)
    cpu_freq_mhz: Optional[float] = None

    # GPU
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    gpu_driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None

    # OS
    os_name: str = ""
    os_version: str = ""
    kernel_version: str = ""


@dataclass
class SoftwareInfo:
    """Software environment information."""
    python_version: str
    pytorch_version: str
    torchvision_version: Optional[str] = None
    transformers_version: Optional[str] = None
    accelerate_version: Optional[str] = None
    numpy_version: Optional[str] = None
    psutil_version: Optional[str] = None


@dataclass
class RuntimeConfig:
    """Runtime configuration for benchmark."""
    device: str
    precision: str
    max_length: int
    batch_size: int
    torch_threads: int
    cuda_visible_devices: Optional[str] = None
    cuda_benchmark: bool = False
    cuda_deterministic: bool = False


@dataclass
class SystemMetadata:
    """Complete system metadata for a benchmark run."""
    hardware: HardwareInfo
    software: SoftwareInfo
    runtime: RuntimeConfig
    run_id: str
    timestamp: str


def collect_hardware_info() -> HardwareInfo:
    """Collect hardware information."""
    # CPU info
    cpu_model = platform.processor() or "Unknown"
    cpu_physical_cores = psutil.cpu_count(logical=False) or 0
    cpu_logical_cores = psutil.cpu_count(logical=True) or 0

    cpu_freq = psutil.cpu_freq()
    cpu_freq_mhz = cpu_freq.current if cpu_freq else None

    # System memory
    mem = psutil.virtual_memory()
    system_ram_total_gb = mem.total / (1024 ** 3)
    system_ram_available_gb = mem.available / (1024 ** 3)

    # GPU info
    gpu_name = None
    gpu_vram_gb = None
    gpu_driver_version = None
    cuda_version = None
    cudnn_version = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_vram_gb = gpu_props.total_memory / (1024 ** 3)

        # CUDA version
        cuda_version = torch.version.cuda

        # cuDNN version
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            cudnn_version = str(torch.backends.cudnn.version())

        # Driver version via nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_driver_version = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # OS info
    os_name = platform.system()
    os_version = platform.version()
    kernel_version = platform.release()

    return HardwareInfo(
        cpu_model=cpu_model,
        cpu_physical_cores=cpu_physical_cores,
        cpu_logical_cores=cpu_logical_cores,
        cpu_freq_mhz=cpu_freq_mhz,
        system_ram_total_gb=system_ram_total_gb,
        system_ram_available_gb=system_ram_available_gb,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        gpu_driver_version=gpu_driver_version,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        os_name=os_name,
        os_version=os_version,
        kernel_version=kernel_version,
    )


def collect_software_info() -> SoftwareInfo:
    """Collect software version information."""
    info = SoftwareInfo(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        pytorch_version=torch.__version__,
    )

    try:
        import torchvision
        info.torchvision_version = torchvision.__version__
    except ImportError:
        pass

    try:
        import transformers
        info.transformers_version = transformers.__version__
    except ImportError:
        pass

    try:
        import accelerate
        info.accelerate_version = accelerate.__version__
    except ImportError:
        pass

    try:
        import numpy
        info.numpy_version = numpy.__version__
    except ImportError:
        pass

    try:
        import psutil
        info.psutil_version = psutil.__version__
    except ImportError:
        pass

    return info


def collect_runtime_config(
    device: str,
    precision: str,
    max_length: int,
    batch_size: int,
) -> RuntimeConfig:
    """Collect runtime configuration."""
    # PyTorch thread settings
    torch_threads = torch.get_num_threads()

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_benchmark = torch.backends.cudnn.benchmark if hasattr(torch.backends, "cudnn") else False
    cuda_deterministic = torch.backends.cudnn.deterministic if hasattr(torch.backends, "cudnn") else False

    return RuntimeConfig(
        device=device,
        precision=precision,
        max_length=max_length,
        batch_size=batch_size,
        torch_threads=torch_threads,
        cuda_visible_devices=cuda_visible,
        cuda_benchmark=cuda_benchmark,
        cuda_deterministic=cuda_deterministic,
    )


def collect_system_metadata(
    run_id: str,
    device: str,
    precision: str,
    max_length: int,
    batch_size: int,
) -> SystemMetadata:
    """Collect complete system metadata for a benchmark run."""
    from datetime import datetime

    hardware = collect_hardware_info()
    software = collect_software_info()
    runtime = collect_runtime_config(device, precision, max_length, batch_size)

    return SystemMetadata(
        hardware=hardware,
        software=software,
        runtime=runtime,
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
    )


def save_metadata(metadata: SystemMetadata, output_dir: str):
    """Save metadata to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "system_info.json")
    with open(filepath, "w") as f:
        json.dump(asdict(metadata), f, indent=2, default=str)
    return filepath


def print_system_summary(metadata: SystemMetadata):
    """Print a formatted system summary."""
    hw = metadata.hardware
    sw = metadata.software
    rt = metadata.runtime

    print("=" * 60)
    print("SYSTEM METADATA")
    print("=" * 60)
    print(f"Run ID: {metadata.run_id}")
    print(f"Timestamp: {metadata.timestamp}")
    print()
    print("HARDWARE:")
    print(f"  CPU: {hw.cpu_model}")
    print(f"  Cores: {hw.cpu_physical_cores} physical, {hw.cpu_logical_cores} logical")
    if hw.cpu_freq_mhz:
        print(f"  CPU Freq: {hw.cpu_freq_mhz:.0f} MHz")
    print(f"  RAM: {hw.system_ram_total_gb:.1f} GB total, {hw.system_ram_available_gb:.1f} GB available")
    if hw.gpu_name:
        print(f"  GPU: {hw.gpu_name}")
        print(f"  VRAM: {hw.gpu_vram_gb:.1f} GB")
        if hw.gpu_driver_version:
            print(f"  Driver: {hw.gpu_driver_version}")
        if hw.cuda_version:
            print(f"  CUDA: {hw.cuda_version}")
        if hw.cudnn_version:
            print(f"  cuDNN: {hw.cudnn_version}")
    print(f"  OS: {hw.os_name} {hw.os_version} ({hw.kernel_version})")
    print()
    print("SOFTWARE:")
    print(f"  Python: {sw.python_version}")
    print(f"  PyTorch: {sw.pytorch_version}")
    if sw.torchvision_version:
        print(f"  TorchVision: {sw.torchvision_version}")
    if sw.transformers_version:
        print(f"  Transformers: {sw.transformers_version}")
    if sw.accelerate_version:
        print(f"  Accelerate: {sw.accelerate_version}")
    print()
    print("RUNTIME:")
    print(f"  Device: {rt.device}")
    print(f"  Precision: {rt.precision}")
    print(f"  Max Length: {rt.max_length}")
    print(f"  Batch Size: {rt.batch_size}")
    print(f"  PyTorch Threads: {rt.torch_threads}")
    if rt.cuda_visible_devices:
        print(f"  CUDA_VISIBLE_DEVICES: {rt.cuda_visible_devices}")
    print(f"  cuDNN Benchmark: {rt.cuda_benchmark}")
    print(f"  cuDNN Deterministic: {rt.cuda_deterministic}")
    print("=" * 60)