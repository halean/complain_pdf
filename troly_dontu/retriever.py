import os
import chromadb

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# initialize client
client = chromadb.PersistentClient(path="chromadb_all_f")
chroma_collection = client.get_collection(name="all_vn_laws")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

"""
Use OpenAI embeddings for retrieval to match the
`all_vn_laws` collection built by open_ai_indexing.py.

Requires OPENAI_API_KEY to be set in the environment.
You can override the model via OPENAI_EMBED_MODEL env var.
"""
embed_model = OpenAIEmbedding(
    model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
)

# load your index from stored vectors
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)

from llama_index.core.retrievers import VectorIndexRetriever

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode


# grouping nodes with same parent
def get_text_from_nodes(nodes):
    parents = {}
    all_text = ""
    
    for node in nodes:
        if NodeRelationship.PARENT in node.node.relationships:
            parents[node.node.relationships[NodeRelationship.PARENT].node_id] = parents.get(
                node.node.relationships[NodeRelationship.PARENT].node_id, []
            )
            parents[node.node.relationships[NodeRelationship.PARENT].node_id].append(node)
    for node in nodes:
        if NodeRelationship.PARENT not in node.node.relationships:
            continue
        if node.node.relationships[NodeRelationship.PARENT].node_id not in parents:
            continue
        text = node.node.relationships[NodeRelationship.PARENT].metadata["text_head"]
        text += "...\n"
        for node in parents[node.node.relationships[NodeRelationship.PARENT].node_id]:
            text += node.node.text
            text += "\n...\n"
        text += f'({node.node.relationships[NodeRelationship.PARENT].metadata["citation"]})'
        text += "\n------------------\n"
        all_text += text
        parents.pop(node.node.relationships[NodeRelationship.PARENT].node_id, None)
    return all_text

async def get_related_regulations(query):
  nodes = await retriever.aretrieve(query)
  return get_text_from_nodes(nodes)
