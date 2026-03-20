"""Model adapters for local LLM inference."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class InferenceResult:
    """Raw output from a model inference call."""

    generated_code: str
    latency_s: float
    error: Optional[str] = None


class ModelAdapter(ABC):
    """Base class for model backends."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        ...


class OllamaAdapter(ModelAdapter):
    """Adapter for Ollama local inference."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.0, max_tokens: int = 2048):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        start = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            latency = time.perf_counter() - start
            raw_text = data.get("response", "")
            code = _extract_code(raw_text)
            return InferenceResult(generated_code=code, latency_s=latency)
        except Exception as exc:
            latency = time.perf_counter() - start
            return InferenceResult(
                generated_code="", latency_s=latency, error=str(exc)
            )


class FakeAdapter(ModelAdapter):
    """Deterministic adapter for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}

    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        code = self.responses.get(prompt, "def solve(*a, **kw): pass")
        return InferenceResult(generated_code=code, latency_s=0.01)


def _extract_code(text: str) -> str:
    """Pull Python code out of an LLM response (strip markdown fences)."""
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    return text.strip()
