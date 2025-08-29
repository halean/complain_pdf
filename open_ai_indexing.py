import re
import pandas as pd

# Define regex patterns for each level
CHUONG_PATTERN = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)", re.IGNORECASE)
DIEU_PATTERN = re.compile(r"^Điều\s+(\d+[a-zđ]?)", re.IGNORECASE)
KHOAN_PATTERN = re.compile(r"^(\d+)\.\s*(.*)")
MUC_PATTERN = re.compile(r"^([a-zđA-ZĐ])\)\s*(.*)")


def parse_text(text):
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

# Load data from a CSV file
data = pd.read_csv("luat.csv")

# Display the first few rows of the dataframe
data = data[
    data["subject"].apply(lambda x: "mới nhất" in x and "sửa đổi" not in x)
]
hierarchical_structures = {}
for subject, text in data[["subject", "text"]].values:
    hierarchical_structures[subject] = parse_text(text)


def get_text_from_structure(dieu, kdm=False):
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

articles = []
documents = []
for law in hierarchical_structures:
    for element in hierarchical_structures[law]:
        if "Chuong" in element:
            # chuong_header = f"Chương {element['Chuong']}: {element['Content']}\n"
            # content = chuong_header
            citation = law
            for dieu in element["Dieu"]:
                article_name = dieu["Dieu"].replace("Dieu", "Điều")
                content = get_text_from_structure(dieu, kdm=False)
                if re.search("Mục [0-9]+[a-zđ]?. ", content):
                    # print(content)
                    content = re.sub("Mục [0-9]+[a-zđ]?. .+?$", "", content)
                if re.search("Luật này (đã )?được Quốc hội", content):
                    content = re.sub(
                        "Luật này (đã )?được Quốc hội.+?$",
                        "",
                        content,
                        flags=re.DOTALL,
                    )
                doc = {
                    "title": article_name,
                    "content": content,
                    "citation": article_name + " " + citation,
                    "law": law,
                }

                articles.append(content)
                documents.append(doc)


from llama_index.core.schema import TextNode

nodes = []
for c in documents:
    citation = c["citation"]
    article_node = TextNode(
        text=c["content"],
        metadata={
            "name": c["title"],
            "type": "Điều",
            "citation": citation,
            "law": c["law"],
        },
    )
    nodes.append(article_node)
# llamaindex nodes with title, title is the first element of chunks, content is the second
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

import chromadb

# index nodes using vectorstore


# Importing ServiceContext is removed due to its deprecation
client = chromadb.PersistentClient(path="./chromadb_all_f")
collection = client.get_or_create_collection(name="all_vn_laws")
# Initialize embedding model: OpenAI text-embedding-3-small
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Initialize vector store (ensure chroma_collection is properly defined)
vector_store = ChromaVectorStore(chroma_collection=collection)
# Initialize storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True,
)
