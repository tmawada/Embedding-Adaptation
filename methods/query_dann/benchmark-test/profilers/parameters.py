"""Parameter counting utilities for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ParameterCounts:
    """Parameter counts for a model configuration."""
    # Encoder (BGE-M3)
    encoder_total: int
    encoder_trainable: int
    encoder_frozen: int

    # Adapter
    adapter_total: int
    adapter_trainable: int

    # Discriminator (training only)
    discriminator_total: int
    discriminator_trainable: int

    # Inference totals
    inference_total_m0: int  # Encoder only (M0)
    inference_total_m1_m2: int  # Encoder + Adapter (M1/M2)

    # Training totals (for Query-DANN)
    training_total: int  # Adapter + Discriminator (encoder frozen)

    # Ratios
    adapter_to_encoder_ratio_pct: float

    # Checkpoint sizes (in bytes, if available)
    encoder_checkpoint_size: Optional[int] = None
    adapter_checkpoint_size: Optional[int] = None
    discriminator_checkpoint_size: Optional[int] = None
    combined_checkpoint_size: Optional[int] = None


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def analyze_model_parameters(
    encoder: torch.nn.Module,
    adapter: Optional[torch.nn.Module] = None,
    discriminator: Optional[torch.nn.Module] = None,
) -> ParameterCounts:
    """Analyze parameter counts for benchmark reporting.

    Args:
        encoder: Frozen encoder (BGE-M3)
        adapter: Query adapter (optional, for M1/M2)
        discriminator: Domain discriminator (optional, training only)

    Returns:
        ParameterCounts with all relevant metrics
    """
    enc_counts = count_parameters(encoder)

    adapter_total = 0
    adapter_trainable = 0
    if adapter is not None:
        adapter_counts = count_parameters(adapter)
        adapter_total = adapter_counts["total"]
        adapter_trainable = adapter_counts["trainable"]

    disc_total = 0
    disc_trainable = 0
    if discriminator is not None:
        disc_counts = count_parameters(discriminator)
        disc_total = disc_counts["total"]
        disc_trainable = disc_counts["trainable"]

    # Inference parameters
    inference_m0 = enc_counts["total"]  # M0: encoder only
    inference_m1_m2 = enc_counts["total"] + adapter_total  # M1/M2: encoder + adapter

    # Training parameters (encoder frozen)
    training_total = adapter_trainable + disc_trainable

    # Ratio
    adapter_ratio = (adapter_total / enc_counts["total"] * 100) if enc_counts["total"] > 0 else 0.0

    return ParameterCounts(
        encoder_total=enc_counts["total"],
        encoder_trainable=enc_counts["trainable"],
        encoder_frozen=enc_counts["frozen"],
        adapter_total=adapter_total,
        adapter_trainable=adapter_trainable,
        discriminator_total=disc_total,
        discriminator_trainable=disc_trainable,
        inference_total_m0=inference_m0,
        inference_total_m1_m2=inference_m1_m2,
        training_total=training_total,
        adapter_to_encoder_ratio_pct=adapter_ratio,
    )


def format_parameter_counts(counts: ParameterCounts) -> str:
    """Format parameter counts for display."""
    lines = [
        "=" * 80,
        "PARAMETER COUNTS",
        "=" * 80,
        f"Encoder (BGE-M3):",
        f"  Total:       {counts.encoder_total:>15,}",
        f"  Trainable:   {counts.encoder_trainable:>15,} (should be 0)",
        f"  Frozen:      {counts.encoder_frozen:>15,}",
        "",
        f"Adapter:",
        f"  Total:       {counts.adapter_total:>15,}",
        f"  Trainable:   {counts.adapter_trainable:>15,}",
        "",
        f"Discriminator (training only):",
        f"  Total:       {counts.discriminator_total:>15,}",
        f"  Trainable:   {counts.discriminator_trainable:>15,}",
        "",
        f"Inference Parameters:",
        f"  M0 (Baseline):     {counts.inference_total_m0:>15,}",
        f"  M1/M2 (+Adapter):  {counts.inference_total_m1_m2:>15,}",
        "",
        f"Training Parameters (Query-DANN):",
        f"  Total (adapter + disc): {counts.training_total:>15,}",
        "",
        f"Adapter/Encoder Ratio: {counts.adapter_to_encoder_ratio_pct:.2f}%",
        "=" * 80,
    ]
    return "\n".join(lines)


def get_checkpoint_sizes(checkpoint_dir: str) -> Dict[str, int]:
    """Get checkpoint file sizes in bytes."""
    import os

    sizes = {}
    files = {
        "encoder_checkpoint_size": "pytorch_model.bin",  # or model.safetensors
        "adapter_checkpoint_size": "adapter.pt",
        "discriminator_checkpoint_size": "discriminator.pt",
    }

    for key, filename in files.items():
        path = os.path.join(checkpoint_dir, filename)
        if os.path.exists(path):
            sizes[key] = os.path.getsize(path)

    # Combined size
    if sizes:
        sizes["combined_checkpoint_size"] = sum(sizes.values())

    return sizes