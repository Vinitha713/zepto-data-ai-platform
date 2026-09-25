
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"

def load_documents():
    documents, ids, metadatas = [], [], []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(text)
            ids.append(path.stem)
            metadatas.append({"source": path.name})
    if len(documents) != 8:
        raise RuntimeError(f"Expected 8 policy documents, found {len(documents)}")
    return documents, ids, metadatas

def build_collection():
    documents, ids, metadatas = load_documents()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return collection

if __name__ == "__main__":
    build_collection()
    print("Indexed 8 Zepto policy documents in ChromaDB.")
