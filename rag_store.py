"""
rag_store.py  -  Embeddings + Chroma + helpers
"""
import chromadb, json
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME  = "arnold_history"      
CHROMA_PATH      = "data/chroma"

# ─────────────────────────────────────────────────────────────────────────────
model      = SentenceTransformer(EMBED_MODEL_NAME)
client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)


def embed(text: str):
    """Devolve um vetor de 384 dimensões"""
    return model.encode(text)


def add_record(rec: dict):
    """Insere UMA interação no índice"""
    doc = f"{rec['user']}\n{rec['assistant']}"
    collection.add(
        ids=[rec["id"]],
        documents=[doc],
        metadatas=[{"ts": rec["ts"]}],
        embeddings=[embed(doc)]
    )


def query_memories(query: str, k: int = 3):
    emb = embed(query)
    out = collection.query(query_embeddings=[emb], n_results=k,
                           include=["documents"])
    return out["documents"][0] if out["documents"] else []
