from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

import requests


class TEIClient:
    """
    Minimal client for Hugging Face Text Embeddings Inference (TEI).

    Supports both native TEI `/embed` and the OpenAI-compatible `/v1/embeddings` route.

    Environment variables:
    - `TEI_BASE_URL`: Base URL to the TEI server (e.g. http://localhost:8080)
    - `TEI_API_KEY`: Optional Bearer token if your TEI sits behind auth
    - `TEI_USE_OPENAI_ROUTE`: If set to truthy, uses `/v1/embeddings`
    - `TEI_MODEL`: Model name to pass when using OpenAI-compatible route
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        use_openai_route: Optional[bool] = None,
        model: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("TEI_BASE_URL") or "").rstrip("/")
        if not self.base_url:
            raise ValueError(
                "TEI base URL is required. Set TEI_BASE_URL or pass base_url."
            )

        self.api_key = api_key or os.getenv("TEI_API_KEY")
        self.timeout = timeout
        if use_openai_route is None:
            env_val = os.getenv("TEI_USE_OPENAI_ROUTE")
            use_openai_route = bool(env_val) and env_val.lower() not in {"0", "false", "no"}
        self.use_openai_route = use_openai_route
        self.model = model or os.getenv("TEI_MODEL")

    # --- Public API -----------------------------------------------------
    def embed(self, inputs: Sequence[str]) -> List[List[float]]:
        """Return embeddings for a batch of input strings.

        Args:
            inputs: Iterable of input texts.

        Returns:
            Nested list of floats with shape (len(inputs), dim).
        """
        texts = list(inputs)
        if not texts:
            return []
        if self.use_openai_route:
            return self._embed_openai_route(texts)
        return self._embed_native(texts)

    def embed_query(self, query: str) -> List[float]:
        embs = self.embed([query])
        return embs[0] if embs else []

    def embed_documents(self, docs: Sequence[str], batch_size: int = 64) -> List[List[float]]:
        # Simple batching helper
        results: List[List[float]] = []
        batch: List[str] = []
        for d in docs:
            batch.append(d)
            if len(batch) >= batch_size:
                results.extend(self.embed(batch))
                batch.clear()
        if batch:
            results.extend(self.embed(batch))
        return results

    # --- Internals ------------------------------------------------------
    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _embed_native(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embed"
        payload = {"inputs": texts}
        r = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Expected shape: {"embeddings": [[...], ...]}
        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]
        # Some deployments may return list directly
        if isinstance(data, list):
            return data
        raise ValueError(f"Unexpected TEI /embed response: {type(data)}")

    def _embed_openai_route(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/v1/embeddings"
        payload = {"input": texts}
        if self.model:
            payload["model"] = self.model
        r = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # OpenAI-compatible: {"data": [{"embedding": [...], ...}, ...]}
        try:
            items = data["data"]
            return [item["embedding"] for item in items]
        except Exception as e:
            raise ValueError(f"Unexpected TEI /v1/embeddings response: {data}") from e


__all__ = ["TEIClient"]

