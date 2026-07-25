import chromadb

from embed_chunks import model
from extract_tables import extract_tables
from extract_text import PDF_PATH

client = chromadb.PersistentClient(path=".chroma")
collection = client.get_or_create_collection("paper_chunks")

if __name__ == "__main__":
    tables = extract_tables(PDF_PATH)
    markdowns = [t["markdown"] for t in tables]
    embeddings = model.encode(markdowns)

    collection.upsert(
        ids=[f"table_{i}" for i in range(len(tables))],
        documents=markdowns,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": PDF_PATH, "type": "table", "page": t["page"]}
            for i, t in enumerate(tables)
        ],
    )

    print(f"Tables stored: {len(tables)}")
    print(f"Total items in paper_chunks collection: {collection.count()}")
