# RAG setup: loads knowledge base, creates FAISS vector index, handles retrieval
# Load .env before initializing client
from dotenv import load_dotenv
load_dotenv()

import faiss
import numpy as np
# Local embeddings — free, no API quota needed
from sentence_transformers import SentenceTransformer

# Load embedding model once
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True).astype("float32")

def load_knowledge_base(path="data/knowledge_base.txt"):
    with open(path, "r") as f:
        content = f.read()
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    return chunks

def build_index(chunks):
    embeddings = np.array([get_embedding(c) for c in chunks])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve(query, chunks, index, top_k=3):
    q_emb = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]]