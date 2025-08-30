#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Optional

import chromadb


def list_collections(client: chromadb.ClientAPI) -> List[str]:
    try:
        cols = client.list_collections()
        return [c.name for c in cols]
    except Exception as e:
        print(f"Error listing collections: {e}", file=sys.stderr)
        return []


def peek_collection(col: chromadb.Collection, k: int = 3):
    try:
        count = col.count()
        print(f"Collection '{col.name}': {count} items")
        if count == 0:
            return
        peek = col.peek(limit=min(k, count))
        ids = peek.get("ids", [])
        docs = peek.get("documents", [])
        metas = peek.get("metadatas", [])
        for i, (id_, doc, meta) in enumerate(zip(ids, docs, metas), start=1):
            print(f"  #{i} id={id_}")
            if meta:
                print(f"     meta: {meta}")
            if doc:
                snippet = doc[:200].replace("\n", " ") + ("…" if len(doc) > 200 else "")
                print(f"     doc : {snippet}")
    except Exception as e:
        print(f"Error peeking collection '{col.name}': {e}", file=sys.stderr)


def resolve_default_model(collection_name: str) -> Optional[str]:
    # Heuristics based on this repo's indexing scripts
    if collection_name == "all_vn_laws_qwen3":
        return os.getenv("QWEN3_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    if collection_name == "all_vn_laws":
        # OpenAI model name; requires OPENAI_API_KEY when used
        return os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    return None


def embed_queries(backend: str, model_name: str, queries: List[str]) -> List[List[float]]:
    if backend == "hf":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        device = os.getenv("QWEN3_DEVICE", "cpu")
        embed = HuggingFaceEmbedding(model_name=model_name, device=device)
        return [embed.get_query_embedding(q) for q in queries]
    elif backend == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        embed = OpenAIEmbedding(model=model_name)
        return [embed.get_query_embedding(q) for q in queries]
    else:
        raise ValueError(f"Unknown backend: {backend}")


def guess_backend(model_name: str) -> str:
    # Very simple heuristic
    if model_name.startswith("text-embedding-") or model_name.startswith("text-embedding-3-"):
        return "openai"
    return "hf"


def run():
    parser = argparse.ArgumentParser(description="Test and query ./chromadb_all_f")
    parser.add_argument(
        "--path",
        default="./chromadb_all_f",
        help="Path to Chroma persistent directory",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name to inspect/query (default: list all)",
    )
    parser.add_argument(
        "--peek",
        type=int,
        default=3,
        help="Number of items to peek from the collection",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional semantic query to run against the collection",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of neighbors to return for query",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model to use for query (auto-detect by collection if omitted)",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "openai"],
        default=None,
        help="Embedding backend (hf or openai). Auto-guessed if omitted.",
    )

    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.path)

    # If no specific collection, list all and exit
    if not args.collection:
        names = list_collections(client)
        if not names:
            print("No collections found.")
            return
        print("Collections found:")
        for n in names:
            print(f"- {n}")
        print("\nUse --collection <name> to inspect or query a collection.")
        return

    # Open collection and peek
    col = client.get_or_create_collection(name=args.collection)
    peek_collection(col, k=args.peek)

    # Optionally run a semantic query
    if args.query:
        model_name = args.model or resolve_default_model(args.collection)
        if not model_name:
            print(
                "Cannot determine default model for this collection. "
                "Provide one via --model.",
                file=sys.stderr,
            )
            sys.exit(2)

        backend = args.backend or guess_backend(model_name)

        print(f"\nRunning query with model='{model_name}' backend='{backend}'…")
        try:
            q_embs = embed_queries(backend, model_name, [args.query])
        except Exception as e:
            print(
                "Failed to load embedding model or compute embeddings.\n"
                "- If using HuggingFace, ensure the model is available locally or network access is allowed.\n"
                "- If using OpenAI, set OPENAI_API_KEY and model name.\n"
                f"Details: {e}",
                file=sys.stderr,
            )
            sys.exit(3)

        try:
            res = col.query(
                query_embeddings=q_embs,
                n_results=max(1, args.top_k),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(
                "Query failed. Ensure the chosen model matches the index's embedding dimension.",
                file=sys.stderr,
            )
            raise

        # Print results
        print("\nTop results:")
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for rank, (id_, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
            print(f"  #{rank} id={id_} distance={dist:.4f}")
            if meta:
                print(f"     meta: {meta}")
            if doc:
                snippet = doc[:400].replace("\n", " ") + ("…" if len(doc) > 400 else "")
                print(f"     doc : {snippet}")


if __name__ == "__main__":
    run()

