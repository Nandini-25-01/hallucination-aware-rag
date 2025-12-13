from loader import load_pdfs
from chunker import chunk_text

text = load_pdfs("data")
chunks = chunk_text(text)

print(f"Total chunks: {len(chunks)}")
print(chunks[0])
