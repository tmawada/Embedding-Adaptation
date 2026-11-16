"""Query-DANN model wrapper for benchmarking.

M1: BGE-M3 + untrained adapter - same frozen encoder followed by a newly initialized, untrained adapter.
M2: BGE-M3 + Query-DANN - same frozen encoder followed by the trained adapter loaded from checkpoint.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, ROOT)

from methods.model import (
    BGEM3Encoder,
    QueryDANN,
    ZeroInitQueryAdapter,
    ADAPTERS,
    GradientReversalLayer,
    SanitizedDomainDiscriminator,
    DomainDiscriminator,
    DISCRIMINATORS,
)


class QueryDANNModel:
    """BGE-M3 encoder + Query-DANN adapter for benchmarking.

    Supports both M1 (untrained adapter) and M2 (trained adapter from checkpoint).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        max_length: int = 512,
        precision: str = "fp32",
        adapter_type: str = "zeroinit",
        discriminator_type: str = "sanitized",
        bottleneck: int = 128,
        adapter_dropout: float = 0.1,
        disc_dropout: float = 0.2,
        checkpoint_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.precision = precision
        self.adapter_type = adapter_type
        self.discriminator_type = discriminator_type
        self.bottleneck = bottleneck
        self.adapter_dropout = adapter_dropout
        self.disc_dropout = disc_dropout
        self.checkpoint_path = checkpoint_path

        # Encoder - always frozen
        self.encoder = BGEM3Encoder(model_name, device=str(self.device), fp16=(precision != "fp32"))

        # Adapter
        self.adapter = ADAPTERS[adapter_type](
            dim=1024, bottleneck=bottleneck, dropout=adapter_dropout
        ).to(self.device)

        # Discriminator (training only, not used at inference)
        self.discriminator = DISCRIMINATORS[discriminator_type](
            dim=1024, dropout=disc_dropout
        ).to(self.device)

        # Gradient reversal layer
        self.grl = GradientReversalLayer()

        # Load checkpoint if provided (M2)
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            # M1: untrained adapter (zero-initialized)
            self._init_untrained()

        # Set to eval mode
        self.adapter.eval()
        self.discriminator.eval()

    def _init_untrained(self):
        """Initialize adapter as untrained (identity-like)."""
        # ZeroInitQueryAdapter already zero-inits the up projection
        # LinearQueryAdapter initializes to identity matrix
        # QueryAdapter (layernorm) zero-inits up projection
        pass  # Initialization is handled in the module constructors

    def _load_checkpoint(self, checkpoint_path: str):
        """Load trained adapter and discriminator from checkpoint."""
        config_path = os.path.join(checkpoint_path, "config.json")
        adapter_path = os.path.join(checkpoint_path, "adapter.pt")
        disc_path = os.path.join(checkpoint_path, "discriminator.pt")

        with open(config_path) as f:
            config = json.load(f)

        # Load adapter state dict
        self.adapter.load_state_dict(torch.load(adapter_path, map_location=self.device))
        # Load discriminator state dict
        self.discriminator.load_state_dict(torch.load(disc_path, map_location=self.device))

    @torch.no_grad()
    def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """Encode texts using BGE-M3 encoder only (no adapter)."""
        emb = self.encoder.encode(texts, max_length=self.max_length, progress=False)
        emb_tensor = torch.from_numpy(emb).to(self.device).float()
        if normalize:
            emb_tensor = F.normalize(emb_tensor, dim=-1)
        return emb_tensor

    @torch.no_grad()
    def adapt(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply adapter to pre-computed embeddings."""
        # Adapter expects float input
        z = self.adapter(embeddings.float())
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def encode_and_adapt(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """Encode texts and apply adapter in one call."""
        z = self.encode(texts, normalize=False)
        return self.adapt(z)

    def get_encoder_params(self) -> int:
        """Get total encoder parameters (frozen)."""
        return sum(p.numel() for p in self.encoder.model.parameters())

    def get_encoder_trainable_params(self) -> int:
        """Get trainable encoder parameters (should be 0 for frozen)."""
        return sum(p.numel() for p in self.encoder.model.parameters() if p.requires_grad)

    def get_adapter_params(self) -> int:
        """Get total adapter parameters."""
        return sum(p.numel() for p in self.adapter.parameters())

    def get_adapter_trainable_params(self) -> int:
        """Get trainable adapter parameters."""
        return sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)

    def get_discriminator_params(self) -> int:
        """Get total discriminator parameters (training only)."""
        return sum(p.numel() for p in self.discriminator.parameters())

    def get_discriminator_trainable_params(self) -> int:
        """Get trainable discriminator parameters."""
        return sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)

    def get_inference_params(self) -> int:
        """Get total inference parameters (encoder + adapter)."""
        return self.get_encoder_params() + self.get_adapter_params()

    def get_training_params(self) -> int:
        """Get total training parameters (adapter + discriminator, encoder frozen)."""
        return self.get_adapter_trainable_params() + self.get_discriminator_trainable_params()

    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        self.encoder.model.to(device)
        self.adapter.to(device)
        self.discriminator.to(device)
        return self

    def eval(self):
        """Set to eval mode."""
        self.encoder.model.eval()
        self.adapter.eval()
        self.discriminator.eval()
        return self


def load_query_dann_model(
    model_name: str = "BAAI/bge-m3",
    device: str = "cuda",
    max_length: int = 512,
    precision: str = "fp32",
    adapter_type: str = "zeroinit",
    discriminator_type: str = "sanitized",
    bottleneck: int = 128,
    adapter_dropout: float = 0.1,
    disc_dropout: float = 0.2,
    checkpoint_path: Optional[str] = None,
) -> QueryDANNModel:
    """Factory function to create Query-DANN model (M1 or M2)."""
    model = QueryDANNModel(
        model_name=model_name,
        device=device,
        max_length=max_length,
        precision=precision,
        adapter_type=adapter_type,
        discriminator_type=discriminator_type,
        bottleneck=bottleneck,
        adapter_dropout=adapter_dropout,
        disc_dropout=disc_dropout,
        checkpoint_path=checkpoint_path,
    )
    model.eval()
    return model