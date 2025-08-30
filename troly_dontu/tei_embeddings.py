from __future__ import annotations

from typing import List, Sequence

# Resolve BaseEmbedding across LlamaIndex versions
try:
    # Newer modular packages expose it here
    from llama_index.core.embeddings import BaseEmbedding  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Direct path in some versions
        from llama_index.core.base.embeddings.base import BaseEmbedding  # type: ignore
    except Exception:
        # Very old monolith fallback (may not exist in this env)
        from llama_index.embeddings.base import BaseEmbedding  # type: ignore

from .tei_client import TEIClient
import asyncio
from pydantic import PrivateAttr


class TEIEmbedding(BaseEmbedding):
    """
    LlamaIndex embedding adapter that uses a TEI server under the hood.

    Example:
        from troly_dontu.tei_embeddings import TEIEmbedding
        embed_model = TEIEmbedding(base_url="http://localhost:8080")
    """

    _client: TEIClient = PrivateAttr()

    def __init__(self, base_url: str, use_openai_route: bool = False, model: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._client = TEIClient(base_url=base_url, use_openai_route=use_openai_route, model=model)

    # LlamaIndex expects these underscored methods
    def _get_query_embedding(self, query: str) -> List[float]:  # type: ignore[override]
        return self._client.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:  # type: ignore[override]
        return await asyncio.to_thread(self._client.embed_query, query)

    def _get_text_embedding(self, text: str) -> List[float]:  # type: ignore[override]
        embs = self._client.embed([text])
        return embs[0] if embs else []

    # Efficient batch implementation used by BaseEmbedding when batching
    def _get_text_embeddings(self, texts: Sequence[str]) -> List[List[float]]:  # type: ignore[override]
        return self._client.embed(list(texts))


__all__ = ["TEIEmbedding"]
