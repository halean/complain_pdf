from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb

from .tei_client import TEIClient


class TEIChromaStore:
    """
    Simple Chroma-based vector store that uses TEI embeddings.

    - Persists to a Chroma collection
    - Computes embeddings via a TEI server using `TEIClient`
    - Supports adding raw texts and querying by text
    """

    def __init__(
        self,
        tei: TEIClient,
        path: str = "./chromadb_all_f",
        collection: str = "tei_default",
        client: Optional[chromadb.ClientAPI] = None,
    ) -> None:
        self.tei = tei
        self.client = client or chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(name=collection)

    # --- Indexing -------------------------------------------------------
    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 64,
    ) -> None:
        if not texts:
            return
        metas: List[Dict[str, Any]] = list(metadatas or [{} for _ in texts])
        if ids is None:
            # Generate simple incremental IDs if not provided
            start = self.col.count()
            ids = [f"tei_{start + i}" for i in range(len(texts))]

        # Compute embeddings in batches to avoid large payloads
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i : i + batch_size])
            chunk_metas = list(metas[i : i + batch_size])
            chunk_ids = list(ids[i : i + batch_size])
            embs = self.tei.embed(chunk)
            self.col.add(
                ids=chunk_ids,
                documents=chunk,
                metadatas=chunk_metas,
                embeddings=embs,
            )

    def add_embeddings(
        self,
        embeddings: Sequence[Sequence[float]],
        documents: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        if ids is None:
            start = self.col.count()
            ids = [f"tei_{start + i}" for i in range(len(embeddings))]
        self.col.add(
            ids=list(ids),
            documents=list(documents) if documents else None,
            metadatas=list(metadatas) if metadatas else None,
            embeddings=[list(e) for e in embeddings],
        )

    # --- Querying -------------------------------------------------------
    def query(
        self,
        text: str,
        top_k: int = 5,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        include = include or ["documents", "metadatas", "distances"]
        q_emb = self.tei.embed_query(text)
        return self.col.query(
            query_embeddings=[q_emb],
            n_results=max(1, int(top_k)),
            include=include,
        )


__all__ = ["TEIChromaStore"]

