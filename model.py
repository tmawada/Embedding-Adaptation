"""Query-DANN components on top of a frozen BGE-M3 dense encoder."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

EMB_DIM = 1024


# --------------------------------------------------------------------------- #
# Frozen base encoder
# --------------------------------------------------------------------------- #
class BGEM3Encoder:
    """Frozen BAAI/bge-m3 dense encoder: [CLS] pooling + L2 normalisation.

    The model is kept in eval mode and every parameter has requires_grad=False,
    so the document (and formal query) representation can never drift.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda", fp16: bool = True):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if fp16 and self.device.type == "cuda" else torch.float32
        try:
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="sdpa")
        except (ValueError, ImportError):
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        max_length: int = 512,
        token_budget: int = 48000,
        out: Optional[np.ndarray] = None,
        desc: str = "encode",
        progress: bool = True,
    ) -> np.ndarray:
        """Encode texts with dynamic, length-sorted batching.

        Batches are sized so that batch_size * longest_sequence <= token_budget.
        Returns (or fills `out`) a float16 array of shape (len(texts), 1024) in the
        original input order.
        """
        if out is None:
            out = np.zeros((len(texts), EMB_DIM), dtype=np.float16)
        # Cheap length proxy for sorting; exact lengths come from the tokenizer per batch.
        order = np.argsort([-len(t) for t in texts], kind="stable")
        pbar = tqdm(total=len(texts), desc=desc, unit="txt", dynamic_ncols=True, disable=not progress)
        i = 0
        while i < len(order):
            longest = self.tokenizer(texts[order[i]], truncation=True, max_length=max_length)["input_ids"]
            bs = max(1, token_budget // len(longest))
            idx = order[i : i + bs]
            try:
                emb = self._encode_batch([texts[j] for j in idx], max_length)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                token_budget = max(512, token_budget // 2)
                continue
            out[idx] = emb
            i += len(idx)
            pbar.update(len(idx))
        pbar.close()
        return out

    def _encode_batch(self, batch: List[str], max_length: int) -> np.ndarray:
        enc = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        cls = self.model(**enc).last_hidden_state[:, 0]
        return F.normalize(cls.float(), dim=-1).half().cpu().numpy()


# --------------------------------------------------------------------------- #
# Gradient reversal
# --------------------------------------------------------------------------- #
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


class GradientReversalLayer(nn.Module):
    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return grad_reverse(x, alpha)


def grl_alpha(progress: float, eta: float = 10.0) -> float:
    """alpha_p = 2 / (1 + exp(-eta * p)) - 1, with p in [0, 1]."""
    return 2.0 / (1.0 + math.exp(-eta * progress)) - 1.0


# --------------------------------------------------------------------------- #
# Query adapter A_psi
# --------------------------------------------------------------------------- #
class QueryAdapter(nn.Module):
    """Residual bottleneck adapter applied to the (L2-normalised) dense query vector.

    z_out = LayerNorm(z_in + Up(Dropout(GELU(LayerNorm(Down(z_in))))))
    """

    def __init__(self, dim: int = EMB_DIM, bottleneck: int = 512, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.norm = nn.LayerNorm(bottleneck)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dim)
        self.out_norm = nn.LayerNorm(dim)
        # Zero-init the up projection so training starts from LayerNorm(z_in),
        # i.e. (close to) the unadapted embedding.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.norm(self.down(z_in))))
        return self.out_norm(z_in + self.up(h))


class ZeroInitQueryAdapter(nn.Module):
    """z_out = L2Norm(z_in + Up(Dropout(GELU(LayerNorm(Down(z_in)))))), Up zero-initialised.

    Inputs are unit-norm, so this is an exact identity at initialisation. Unlike a
    final LayerNorm, it does not force every output to zero component mean, which
    gave the discriminator a trivial 100%-accurate shortcut in the first run.
    """

    def __init__(self, dim: int = EMB_DIM, bottleneck: int = 128, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.norm = nn.LayerNorm(bottleneck)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.norm(self.down(z_in))))
        return F.normalize(z_in + self.up(h), dim=-1)


# --------------------------------------------------------------------------- #
# Domain discriminator D_phi
# --------------------------------------------------------------------------- #
class DomainDiscriminator(nn.Module):
    """1024 -> 512 -> 256 -> 1. Returns *logits*; apply sigmoid for probabilities.

    The final Sigmoid is folded into BCEWithLogitsLoss, which is numerically
    stable and autocast-safe (nn.BCELoss is rejected under mixed precision).
    """

    def __init__(self, dim: int = EMB_DIM, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(z))


class SanitizedDomainDiscriminator(DomainDiscriminator):
    """Centres each vector to zero component mean and re-normalises before classifying,
    so the domain decision can only use direction, not first-order statistics
    (component mean, norm) that output normalisation layers can leak."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z - z.mean(dim=-1, keepdim=True), dim=-1)
        return super().forward(z)


class IdentityQueryAdapter(nn.Module):
    """No adaptation at all (ablation: --use_adapter false).

    Has no parameters, so with a frozen encoder nothing on the query side is trainable and the
    pipeline reduces exactly to the base model. Useful as a lower bound / plumbing check.
    """

    def __init__(self, dim: int = EMB_DIM, bottleneck: int = 0, dropout: float = 0.0):
        super().__init__()

    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        return F.normalize(z_in, dim=-1)


class LinearQueryAdapter(nn.Module):
    """Plain 1024x1024 linear map, initialised to the identity (ablation: --adapter linear).

    Same interface as the bottleneck adapter but without the bottleneck, residual branch,
    LayerNorm, GELU or dropout: isolates what the adapter architecture itself contributes.
    """

    def __init__(self, dim: int = EMB_DIM, bottleneck: int = 0, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(z_in), dim=-1)


ADAPTERS = {"zeroinit": ZeroInitQueryAdapter, "layernorm": QueryAdapter,
            "linear": LinearQueryAdapter, "identity": IdentityQueryAdapter}
DISCRIMINATORS = {"sanitized": SanitizedDomainDiscriminator, "plain": DomainDiscriminator}


class QueryDANN(nn.Module):
    """Trainable head: adapter + GRL + discriminator (the encoder lives outside).

    adapter="layernorm", discriminator="plain", bottleneck=512 reproduces the first run.
    """

    def __init__(self, bottleneck: int = 128, adapter_dropout: float = 0.1, disc_dropout: float = 0.2,
                 adapter: str = "zeroinit", discriminator: str = "sanitized"):
        super().__init__()
        self.adapter = ADAPTERS[adapter](EMB_DIM, bottleneck, adapter_dropout)
        self.grl = GradientReversalLayer()
        self.discriminator = DISCRIMINATORS[discriminator](EMB_DIM, disc_dropout)

    def adapt(self, z_query: torch.Tensor) -> torch.Tensor:
        """Adapted, L2-normalised query embedding (cosine space shared with docs)."""
        return F.normalize(self.adapter(z_query).float(), dim=-1)

    def domain_logits(self, z: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.discriminator(self.grl(z, alpha))
