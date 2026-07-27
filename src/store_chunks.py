import chromadb

from chunk_text import splitter
from embed_chunks import model
from extract_text import PDF_PATH, extract_text

client = chromadb.PersistentClient(path=".chroma")
collection = client.get_or_create_collection("document_chunks")

if __name__ == "__main__":
    full_text = extract_text(PDF_PATH)
    chunks = splitter.split_text(full_text)
    embeddings = model.encode(chunks)

    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": PDF_PATH, "chunk_index": i, "type": "text"} for i in range(len(chunks))
        ],
    )

    print(f"Chunks stored: {collection.count()}")
    sample = collection.get(ids=["chunk_0"], include=["documents", "metadatas"])
    print("--- Sample stored record (chunk_0) ---")
    print(sample["documents"][0][:150])
    print(sample["metadatas"][0])
