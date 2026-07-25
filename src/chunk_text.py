from langchain_text_splitters import RecursiveCharacterTextSplitter

from extract_text import PDF_PATH, extract_text

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

if __name__ == "__main__":
    full_text = extract_text(PDF_PATH)
    chunks = splitter.split_text(full_text)

    print(f"Total chunks: {len(chunks)}")
    print("--- Chunk 0 ---")
    print(chunks[0])
    print("--- Chunk 1 ---")
    print(chunks[1])

    with open("data/chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i} ---\n{chunk}\n\n")
