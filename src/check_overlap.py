from chunk_text import splitter
from extract_text import PDF_PATH, extract_text

full_text = extract_text(PDF_PATH)
chunks = splitter.split_text(full_text)


def longest_common_suffix_prefix(a: str, b: str) -> str:
    max_len = min(len(a), len(b))
    for length in range(max_len, 0, -1):
        if a[-length:] == b[:length]:
            return a[-length:]
    return ""


for i in range(3):
    overlap = longest_common_suffix_prefix(chunks[i], chunks[i + 1])
    print(f"Chunk {i} -> Chunk {i+1}: overlap length = {len(overlap)}")
    print(f"  Overlap text: {overlap!r}")
