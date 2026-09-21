import json
import time
from typing import Generator
import requests


class OllamaClient:
    """
    Client for interacting with local Ollama instance.

    Configured for low-latency Streaming Live RAG:
    - think=False suppresses model internal chain-of-thought generation
    - temperature=0.0 and num_predict for concise, deterministic output
    - Measures TTFT (time-to-first-token) and total generation latency
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        think: bool = False,
        temperature: float = 0.0,
        num_predict: int = 300,
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url
        self.think = think
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

        self.last_latency_ms: float = 0.0
        self.last_ttft_ms: float = 0.0

        # Token accounting exactly as reported by Ollama's own response
        # fields (prompt_eval_count / eval_count). None = not reported.
        # These are never estimated.
        self.last_prompt_tokens: int | None = None
        self.last_completion_tokens: int | None = None

    def _reset_usage(self) -> None:
        self.last_prompt_tokens = None
        self.last_completion_tokens = None

    def _record_usage(self, data: dict) -> None:
        """Capture Ollama-reported token counts from a final /api/generate payload."""
        prompt = data.get("prompt_eval_count")
        completion = data.get("eval_count")
        self.last_prompt_tokens = prompt if isinstance(prompt, int) else None
        self.last_completion_tokens = (
            completion if isinstance(completion, int) else None
        )

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }

        self._reset_usage()
        start_time = time.perf_counter()

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        total_time = (time.perf_counter() - start_time) * 1000.0
        self.last_latency_ms = total_time
        self.last_ttft_ms = total_time

        data = response.json()
        self._record_usage(data)

        return data["response"]

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }

        self._reset_usage()
        start_time = time.perf_counter()
        ttft_recorded = False

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=(10, self.timeout),
        )

        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            data = line.decode("utf-8")
            chunk_data = json.loads(data)

            token = chunk_data.get("response", "")

            if token:
                if not ttft_recorded:
                    self.last_ttft_ms = (time.perf_counter() - start_time) * 1000.0
                    ttft_recorded = True
                yield token

            if chunk_data.get("done", False):
                self.last_latency_ms = (time.perf_counter() - start_time) * 1000.0
                self._record_usage(chunk_data)
                break

    def get_last_metrics(self) -> dict[str, float | int | None]:
        prompt = self.last_prompt_tokens
        completion = self.last_completion_tokens

        return {
            "ttft_ms": round(self.last_ttft_ms, 2),
            "latency_ms": round(self.last_latency_ms, 2),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": (
                prompt + completion
                if prompt is not None and completion is not None
                else None
            ),
        }