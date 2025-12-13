from pypdf import PdfReader
import os

def load_pdfs(folder_path: str) -> str:
    full_text = ""

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

    return full_text


if __name__ == "__main__":
    text = load_pdfs("data")
    print(text[:1000])
