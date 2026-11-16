"""Ollama HTTP client wrapper with latency tracking and error handling."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GenerationResult:
    """Result from a generation request."""
    text: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    done: bool = True


@dataclass
class ModelInfo:
    """Model information from Ollama."""
    name: str
    size: int
    digest: str
    details: Dict[str, Any]


class OllamaClient:
    """Synchronous Ollama client with latency tracking."""

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "qwen2.5:3b-instruct-q4_K_M",
        timeout: int = 120,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session_headers = {"Content-Type": "application/json"}

    def _request(
        self,
        path: str,
        payload: Dict[str, Any],
        stream: bool = False,
    ) -> urllib.request.urlopen:
        """Make HTTP request to Ollama API."""
        url = f"{self.endpoint}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers=self._session_headers,
        )
        return urllib.request.urlopen(req, timeout=self.timeout)

    def generate(
        self,
        prompt: str,
        system: str = "",
        template: str = "",
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        raw: bool = False,
    ) -> GenerationResult:
        """Generate text from prompt with latency tracking."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "raw": raw,
        }
        if system:
            payload["system"] = system
        if template:
            payload["template"] = template
        if options:
            payload["options"] = options

        t0 = time.monotonic()
        try:
            with self._request("/api/generate", payload, stream=stream) as response:
                if stream:
                    # Handle streaming response
                    full_text = ""
                    prompt_tokens = 0
                    completion_tokens = 0
                    for line in response:
                        line = line.decode("utf-8").strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        if "response" in data:
                            full_text += data["response"]
                        if "prompt_eval_count" in data:
                            prompt_tokens = data["prompt_eval_count"]
                        if "eval_count" in data:
                            completion_tokens = data["eval_count"]
                        if data.get("done", False):
                            break
                    latency = time.monotonic() - t0
                    return GenerationResult(
                        text=full_text.strip(),
                        latency_s=latency,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        model=self.model,
                        done=True,
                    )
                else:
                    # Non-streaming response
                    data = json.load(response)
                    latency = time.monotonic() - t0
                    return GenerationResult(
                        text=data.get("response", "").strip(),
                        latency_s=latency,
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                        total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                        model=self.model,
                        done=data.get("done", True),
                    )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            raise RuntimeError(f"Ollama API error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error: {e.reason}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> GenerationResult:
        """Chat completion with messages format."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options

        t0 = time.monotonic()
        try:
            with self._request("/api/chat", payload, stream=stream) as response:
                if stream:
                    full_text = ""
                    prompt_tokens = 0
                    completion_tokens = 0
                    for line in response:
                        line = line.decode("utf-8").strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            full_text += data["message"]["content"]
                        if "prompt_eval_count" in data:
                            prompt_tokens = data["prompt_eval_count"]
                        if "eval_count" in data:
                            completion_tokens = data["eval_count"]
                        if data.get("done", False):
                            break
                    latency = time.monotonic() - t0
                    return GenerationResult(
                        text=full_text.strip(),
                        latency_s=latency,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        model=self.model,
                        done=True,
                    )
                else:
                    data = json.load(response)
                    latency = time.monotonic() - t0
                    msg = data.get("message", {})
                    return GenerationResult(
                        text=msg.get("content", "").strip(),
                        latency_s=latency,
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                        total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                        model=self.model,
                        done=data.get("done", True),
                    )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            raise RuntimeError(f"Ollama API error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error: {e.reason}")

    def list_models(self) -> List[ModelInfo]:
        """List available models."""
        url = f"{self.endpoint}/api/tags"
        req = urllib.request.Request(url, method="GET", headers=self._session_headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = json.load(response)
        return [
            ModelInfo(
                name=m["name"],
                size=m.get("size", 0),
                digest=m.get("digest", ""),
                details=m.get("details", {}),
            )
            for m in data.get("models", [])
        ]

    def show_model(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Show model details."""
        model_name = name or self.model
        payload = {"name": model_name}
        with self._request("/api/show", payload) as response:
            return json.load(response)

    def pull_model(self, name: str, stream: bool = True) -> None:
        """Pull a model from Ollama registry."""
        payload = {"name": name, "stream": stream}
        with self._request("/api/pull", payload, stream=stream) as response:
            if stream:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if line:
                        data = json.loads(line)
                        if "status" in data:
                            print(f"  {data['status']}", end="\r")
                print()  # Newline after progress
            else:
                json.load(response)

    def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            with urllib.request.urlopen(f"{self.endpoint}/api/tags", timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    @classmethod
    def from_config(cls, config_path: str) -> "OllamaClient":
        """Create client from YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_cfg = config.get("model", {})
        return cls(
            endpoint=model_cfg.get("endpoint", "http://localhost:11434"),
            model=model_cfg.get("name", "qwen2.5:3b-instruct-q4_K_M"),
        )


class OllamaClientAsync:
    """Async Ollama client (requires aiohttp)."""
    
    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "qwen2.5:3b-instruct-q4_K_M",
        timeout: int = 120,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._session

    async def generate(
        self,
        prompt: str,
        system: str = "",
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> GenerationResult:
        """Async generate text from prompt."""
        session = await self._get_session()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        t0 = time.monotonic()
        try:
            async with session.post(f"{self.endpoint}/api/generate", json=payload) as response:
                if stream:
                    full_text = ""
                    prompt_tokens = 0
                    completion_tokens = 0
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        if "response" in data:
                            full_text += data["response"]
                        if "prompt_eval_count" in data:
                            prompt_tokens = data["prompt_eval_count"]
                        if "eval_count" in data:
                            completion_tokens = data["eval_count"]
                        if data.get("done", False):
                            break
                    latency = time.monotonic() - t0
                    return GenerationResult(
                        text=full_text.strip(),
                        latency_s=latency,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        model=self.model,
                        done=True,
                    )
                else:
                    data = await response.json()
                    latency = time.monotonic() - t0
                    return GenerationResult(
                        text=data.get("response", "").strip(),
                        latency_s=latency,
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                        total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                        model=self.model,
                        done=data.get("done", True),
                    )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()