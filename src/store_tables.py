import chromadb

from embed_chunks import model
from extract_tables import extract_tables
from extract_text import PDF_PATH

client = chromadb.PersistentClient(path=".chroma")
collection = client.get_or_create_collection("document_chunks")

if __name__ == "__main__":
    tables = extract_tables(PDF_PATH)

    # Store the caption alongside the grid so the LLM knows what the table represents,
    # and embed caption-first so questions match the table's natural-language description
    # rather than a wall of pipes and digits. Without this, stored tables never ranked in
    # the top-3 for any evaluation question.
    documents = [f"{t['caption']}\n\n{t['markdown']}" for t in tables]
    embeddings = model.encode(documents)

    collection.upsert(
        ids=[f"table_{i}" for i in range(len(tables))],
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": PDF_PATH, "type": "table", "page": t["page"], "caption": t["caption"]}
            for t in tables
        ],
    )

    print(f"Tables stored: {len(tables)}")
    for i, t in enumerate(tables):
        print(f"  table_{i}: {t['caption'][:90]}")
    print(f"Total items in document_chunks collection: {collection.count()}")
