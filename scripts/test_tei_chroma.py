#!/usr/bin/env python3
import argparse
import os
from typing import List

from troly_dontu.tei_client import TEIClient
from troly_dontu.tei_vector_store import TEIChromaStore


def run():
    parser = argparse.ArgumentParser(description="Test TEI + Chroma vector store")
    parser.add_argument("--base-url", default=os.getenv("TEI_BASE_URL"), help="TEI base URL")
    parser.add_argument("--use-openai-route", action="store_true", help="Use /v1/embeddings route")
    parser.add_argument("--model", default=os.getenv("TEI_MODEL"), help="Model name for /v1/embeddings route")
    parser.add_argument("--path", default="./chromadb_all_f", help="Chroma persistence path")
    parser.add_argument("--collection", default="tei_demo", help="Chroma collection name")
    parser.add_argument("--index", nargs="*", default=None, help="Optional texts to index")
    parser.add_argument("--query", default=None, help="Query text to run")
    parser.add_argument("--top-k", type=int, default=5, help="Neighbors to return")
    args = parser.parse_args()

    if not args.base_url:
        raise SystemExit("Provide TEI base URL via --base-url or TEI_BASE_URL env var")

    tei = TEIClient(
        base_url=args.base_url,
        use_openai_route=args.use_openai_route,
        model=args.model,
    )
    store = TEIChromaStore(tei, path=args.path, collection=args.collection)

    if args.index:
        print(f"Indexing {len(args.index)} texts into collection '{args.collection}'…")
        store.add_texts(args.index)

    if args.query:
        print(f"\nQuery: {args.query}")
        res = store.query(args.query, top_k=args.top_k)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, (id_, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
            snippet = (doc or "")[:200].replace("\n", " ")
            print(f"  #{i} id={id_} dist={dist:.4f} meta={meta} doc={snippet}")


if __name__ == "__main__":
    run()

