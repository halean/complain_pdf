# Complain PDF – Embeddings + Chroma + TEI

End‑to‑end scripts and helpers to parse Vietnamese law texts, build vector indexes in Chroma, and retrieve passages using OpenAI, Hugging Face, or a Hugging Face TEI (Text Embeddings Inference) server.

## What's Inside

- Indexing
  - `open_ai_indexing.py`: Build Chroma collection with OpenAI embeddings.
  - `qwen3_indexing.py`: Build Chroma collection with HuggingFace Qwen3 embeddings.
  - `tei_indexing.py`: Build Chroma collection using a TEI server for embeddings.
- Retrieval
  - `troly_dontu/retriever.py`: LlamaIndex retriever wired to Chroma + OpenAI embeddings.
- TEI Support
  - `troly_dontu/tei_client.py`: Minimal TEI client (native `/embed` or OpenAI‑compatible `/v1/embeddings`).
  - `troly_dontu/tei_embeddings.py`: LlamaIndex `BaseEmbedding` adapter backed by TEI.
  - `troly_dontu/tei_vector_store.py`: Simple Chroma wrapper that uses TEI embeddings.
- Utilities
  - `scripts/test_chromadb_all_f.py`: Inspect/query existing Chroma collections.
  - `scripts/test_tei_chroma.py`: Quick demo to index/query with TEI + Chroma.

Input data is expected in `luat.csv` with columns: `subject`, `text`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```

## Environment

- OpenAI
  - `OPENAI_API_KEY`: API key for OpenAI
  - `OPENAI_EMBED_MODEL` (optional): default `text-embedding-3-small`
- Hugging Face (Qwen3)
  - `QWEN3_EMBED_MODEL` (optional): default `Qwen/Qwen3-Embedding-0.6B`
  - `QWEN3_DEVICE` (optional): e.g., `cpu`
- TEI (Text Embeddings Inference)
  - `TEI_BASE_URL`: e.g., `http://localhost:8080`
  - `TEI_USE_OPENAI_ROUTE` (optional): `1/true` to call `/v1/embeddings` instead of `/embed`
  - `TEI_MODEL` (optional): model name used with the OpenAI‑compatible route
  - `TEI_API_KEY` (optional): Bearer token if your TEI is behind auth

## Build an Index

OpenAI

```bash
export OPENAI_API_KEY=...
python open_ai_indexing.py
```

Qwen3 (HuggingFace locally)

```bash
python qwen3_indexing.py
```

TEI server

```bash
export TEI_BASE_URL=http://localhost:8080
# optional: export TEI_USE_OPENAI_ROUTE=1

python tei_indexing.py
```

Each script persists to `./chromadb_all_f` with distinct collections:

- OpenAI: `all_vn_laws`
- Qwen3: `all_vn_laws_qwen3`
- TEI: `all_vn_laws_tei`

## Query Existing Collections

Generic inspector + semantic query:

```bash
python scripts/test_chromadb_all_f.py                    # list collections
python scripts/test_chromadb_all_f.py --collection all_vn_laws --peek 3

# Run a query (auto‑selects backend by model name)
python scripts/test_chromadb_all_f.py --collection all_vn_laws --query "thủ tục nộp thuế" --top-k 5
```

TEI demo end‑to‑end:

```bash
export TEI_BASE_URL=http://localhost:8080
python scripts/test_tei_chroma.py --index "xin chào" "chào thế giới" --query "lời chào" --top-k 3
```

## Use TEI in LlamaIndex

```python
from troly_dontu.tei_embeddings import TEIEmbedding
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

embed_model = TEIEmbedding(base_url="http://localhost:8080")
client = chromadb.PersistentClient(path="./chromadb_all_f")
collection = client.get_or_create_collection(name="tei_collection")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model, show_progress=True)
```

## File Map

- `open_ai_indexing.py`
- `qwen3_indexing.py`
- `tei_indexing.py`
- `troly_dontu/retriever.py`
- `troly_dontu/tei_client.py`
- `troly_dontu/tei_embeddings.py`
- `troly_dontu/tei_vector_store.py`
- `scripts/test_chromadb_all_f.py`
- `scripts/test_tei_chroma.py`

## Troubleshooting

- LlamaIndex import paths differ by version. `tei_embeddings.py` includes a compatibility shim for `BaseEmbedding`.
- If `show_progress=True` is used, progress bars are handled by LlamaIndex; the adapter batches via `_get_text_embeddings`.
- TEI requests fail: ensure `TEI_BASE_URL` is reachable and matches the route you enable (native `/embed` vs `/v1/embeddings`).

