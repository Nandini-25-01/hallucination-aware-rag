from loader import load_pdfs
from chunker import chunk_text
from embeddings import embed_texts
from vector_store import build_faiss_index, search_index


text = load_pdfs("data")
chunks = chunk_text(text)
chunk_embeddings = embed_texts(chunks)
index = build_faiss_index(chunk_embeddings)
query = "How many leaves are employees allowed?"
query_embedding = embed_texts([query])[0]

indices, distances = search_index(index, query_embedding)
print("Top matching chunks:")
for i in indices:
    print("-" * 40)
    print(chunks[i][:300])
