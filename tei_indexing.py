import os
import re
from typing import Dict, List

import pandas as pd
import chromadb

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from troly_dontu.tei_embeddings import TEIEmbedding


# ---------- Load and parse source CSV ----------
# Expects a CSV file "luat.csv" with columns: subject, text
data = pd.read_csv("luat.csv")
data = data[data["subject"].apply(lambda x: "mới nhất" in x and "sửa đổi" not in x)]

# Define regex patterns for each level
CHUONG_PATTERN = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)", re.IGNORECASE)
DIEU_PATTERN = re.compile(r"^Điều\s+(\d+[a-zđ]?)", re.IGNORECASE)
KHOAN_PATTERN = re.compile(r"^(\d+)\.\s*(.*)")
MUC_PATTERN = re.compile(r"^([a-zđA-ZĐ])\)\s*(.*)")


def parse_text(text: str):
    hierarchy = []
    current_chuong = None
    current_dieu = None
    current_khoan = None
    current_muc = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Check for Chương
        chuong_match = CHUONG_PATTERN.match(line)
        if chuong_match:
            chuong_title = line
            current_chuong = {"Chuong": chuong_title, "Dieu": []}
            hierarchy.append(current_chuong)
            current_dieu = None
            current_khoan = None
            current_muc = None
            continue

        # Check for Điều
        dieu_match = DIEU_PATTERN.match(line)
        if dieu_match:
            dieu_number = dieu_match.group(1)
            dieu_content = line[dieu_match.end() :].strip(". ").strip()
            current_dieu = {
                "Dieu": f"Dieu {dieu_number}",
                "Content": dieu_content,
                "Khoan": [],
            }
            if current_chuong is None:
                # If no Chương, create a default one
                current_chuong = {"Chuong": "Chuong 0", "Dieu": []}
                hierarchy.append(current_chuong)
            current_chuong["Dieu"].append(current_dieu)
            current_khoan = None
            current_muc = None
            continue

        # Check for Khoản
        khoan_match = KHOAN_PATTERN.match(line)
        if khoan_match and current_dieu is not None:
            khoan_number = khoan_match.group(1)
            khoan_content = khoan_match.group(2)
            current_khoan = {
                "Khoan": f"Khoan {khoan_number}",
                "Content": khoan_content,
                "Muc": [],
            }
            current_dieu["Khoan"].append(current_khoan)
            current_muc = None
            continue

        # Check for Mục
        muc_match = MUC_PATTERN.match(line)
        if muc_match and current_khoan is not None:
            muc_letter = muc_match.group(1)
            muc_content = muc_match.group(2)
            current_muc = {
                "Muc": f"Muc {muc_letter}",
                "Content": muc_content,
            }
            current_khoan["Muc"].append(current_muc)
            continue

        # If none of the above, it's part of the current content
        if current_muc:
            current_muc["Content"] += " " + line
        elif current_khoan:
            current_khoan["Content"] += " " + line
        elif current_dieu:
            current_dieu["Content"] += " " + line
        elif current_chuong:
            current_chuong["Content"] = (
                current_chuong.get("Content", "") + " " + line
            )
        else:
            # Content before any Chương or Điều
            if "Preamble" not in hierarchy:
                hierarchy.append({"Preamble": line})
            else:
                hierarchy[-1]["Preamble"] += " " + line

    return hierarchy


hierarchical_structures: Dict[str, List[Dict]] = {}
for subject, text in data[["subject", "text"]].values:
    hierarchical_structures[subject] = parse_text(text)


def get_text_from_structure(dieu: Dict, kdm: bool = False) -> str:
    head = dieu["Dieu"].replace("Dieu", "Điều")
    head = f"{head}. {dieu['Content']}\n"
    content = f"{head}"
    clause_annotation = ""
    item_annotation = ""
    clause_prefix = ""
    item_prefix = ""
    if kdm:
        clause_annotation = "Khoản "
        item_annotation = "Điểm "
        clause_prefix = dieu["Dieu"].replace("Dieu", "Điều") + " "
    for khoan in dieu["Khoan"]:
        content += f"{khoan['Khoan'].replace('Khoan ',clause_annotation)}. {clause_prefix}{khoan['Content']}\n"
        if kdm:
            item_prefix = f"{khoan['Khoan'].replace('Khoan ',clause_annotation)} {clause_prefix} "
        if len(khoan["Muc"]) > 0:
            for muc in khoan["Muc"]:
                content += f"{muc['Muc'].replace('Muc ',item_annotation)}) {item_prefix}{muc['Content']}\n"
    return content


# ---------- Build nodes ----------
documents: List[Dict] = []
for law in hierarchical_structures:
    for element in hierarchical_structures[law]:
        if "Chuong" in element:
            citation = law
            for dieu in element["Dieu"]:
                article_name = dieu["Dieu"].replace("Dieu", "Điều")
                content = get_text_from_structure(dieu, kdm=False)
                if re.search("Mục [0-9]+[a-zđ]?. ", content):
                    content = re.sub("Mục [0-9]+[a-zđ]?. .+?$", "", content)
                if re.search("Luật này (đã )?được Quốc hội", content):
                    content = re.sub(
                        "Luật này (đã )?được Quốc hội.+?$", "", content, flags=re.DOTALL
                    )
                documents.append(
                    {
                        "title": article_name,
                        "content": content,
                        "citation": article_name + " " + citation,
                        "law": law,
                    }
                )


nodes: List[TextNode] = []
for c in documents:
    nodes.append(
        TextNode(
            text=c["content"],
            metadata={
                "name": c["title"],
                "type": "Điều",
                "citation": c["citation"],
                "law": c["law"],
            },
        )
    )


# ---------- Vector store + TEI embedding ----------
client = chromadb.PersistentClient(path="./chromadb_all_f")
# Keep a separate collection to avoid mixing embeddings from different models
collection = client.get_or_create_collection(name="all_vn_laws_tei")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# TEI base URL; default to localhost if not provided
tei_base_url = os.getenv("TEI_BASE_URL", "http://localhost:8080")
# Use OpenAI-compatible route if env set (e.g., TEI_USE_OPENAI_ROUTE=1)
use_openai_route = os.getenv("TEI_USE_OPENAI_ROUTE", "0").lower() not in {"0", "false", "no", ""}
tei_model = os.getenv("TEI_MODEL")

embed_model = TEIEmbedding(
    base_url=tei_base_url,
    use_openai_route=use_openai_route,
    model=tei_model,
)


# ---------- Build index ----------
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True,
)

