"""Baseline BGE-M3 model wrapper for benchmarking.

M0: BGE-M3 baseline - original BAAI/bge-m3, no adapter.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, ROOT)

from methods.model import BGEM3Encoder


class BaselineModel:
    """Frozen BGE-M3 encoder for baseline (M0) benchmarking."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        max_length: int = 512,
        precision: str = "fp32",
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.precision = precision

        # Use float32 for primary comparison as per spec
        dtype = torch.float32 if precision == "fp32" else torch.float16

        self.encoder = BGEM3Encoder(model_name, device=str(self.device), fp16=(precision != "fp32"))

    def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """Encode texts to embeddings using frozen BGE-M3."""
        with torch.no_grad():
            # BGEM3Encoder returns numpy array, convert to tensor
            emb = self.encoder.encode(texts, max_length=self.max_length, progress=False)
            emb_tensor = torch.from_numpy(emb).to(self.device).float()
            if normalize:
                emb_tensor = F.normalize(emb_tensor, dim=-1)
            return emb_tensor

    def get_encoder_params(self) -> int:
        """Get total encoder parameters."""
        return sum(p.numel() for p in self.encoder.model.parameters())

    def get_encoder_trainable_params(self) -> int:
        """Get trainable encoder parameters (should be 0 for frozen)."""
        return sum(p.numel() for p in self.encoder.model.parameters() if p.requires_grad)

    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        self.encoder.model.to(device)
        return self

    def eval(self):
        """Set to eval mode."""
        self.encoder.model.eval()
        return self


def load_baseline_model(
    model_name: str = "BAAI/bge-m3",
    device: str = "cuda",
    max_length: int = 512,
    precision: str = "fp32",
) -> BaselineModel:
    """Factory function to create baseline model."""
    model = BaselineModel(model_name, device, max_length, precision)
    model.eval()
    return model