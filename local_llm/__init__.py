"""Local LLM package for query normalization using Ollama."""

from .client import OllamaClient, OllamaClientAsync
from .normalize import LocalLLMNormalizer

__all__ = [
    "OllamaClient",
    "OllamaClientAsync",
    "LocalLLMNormalizer",
]

__version__ = "0.1.0"