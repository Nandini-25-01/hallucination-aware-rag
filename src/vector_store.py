import faiss
import numpy as np

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]  # embedding dimension
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, k=5):
    distances, indices = index.search(
        np.array([query_embedding]), k
    )
    return indices[0], distances[0]
