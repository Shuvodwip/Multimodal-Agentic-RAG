from sentence_transformers import SentenceTransformer

from chunk_text import splitter
from extract_text import PDF_PATH, extract_text

model = SentenceTransformer("all-MiniLM-L6-v2")

if __name__ == "__main__":
    full_text = extract_text(PDF_PATH)
    chunks = splitter.split_text(full_text)

    embeddings = model.encode(chunks)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First embedding, first 5 values: {embeddings[0][:5]}")
