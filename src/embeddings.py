from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]):
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings
