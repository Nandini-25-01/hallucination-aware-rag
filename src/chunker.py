from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_pdfs

def chunk_text(text: str, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    print("Chunker started")

    text = load_pdfs("data")
    print("Total characters loaded:", len(text))

    chunks = chunk_text(text)

    print("Number of chunks:", len(chunks))
    print("\n--- Sample Chunk ---\n")
    print(chunks[0][:500])
