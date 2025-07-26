from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

def load_cleaned_text(path="data/cleaned_text.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "।", "."]
    )
    return splitter.split_text(text)

def embed_chunks(chunks):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(chunks)
    return embeddings

def save_faiss_index(embeddings, chunks):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("data/faiss", exist_ok=True)
    faiss.write_index(index, "data/faiss/index.bin")

    with open("data/faiss/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ FAISS index and chunks saved.")

if __name__ == "__main__":
    text = load_cleaned_text()
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    save_faiss_index(embeddings, chunks)
