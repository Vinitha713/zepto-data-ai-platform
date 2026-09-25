
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"

_model = None

def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_collection(COLLECTION_NAME)

def retrieve(query: str, k: int = 3):
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = get_collection()
    result = collection.query(
        query_embeddings=[_model.encode(query, normalize_embeddings=True).tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {"id": ids[i], "document": docs[i], "source": metas[i].get("source", ids[i]),
         "distance": distances[i]}
        for i in range(len(docs))
    ]
