"""Stable, namespaced identifiers for stored passages.

Ids used to be positional (`chunk_0`, `table_0`), which is fine for one document and
broken for several: the second document's `chunk_0` silently overwrites the first's.
Prefixing with a per-document key keeps them distinct, while re-uploading the same
filename deliberately reuses its key so the document is refreshed rather than
duplicated.
"""

import hashlib
import os
import re

SEPARATOR = "__"

# e.g. "a1b2c3d4__chunk_12". The document key is hex, so the pattern stays unambiguous.
ID_PATTERN = re.compile(rf"([0-9a-f]{{8}}){SEPARATOR}((?:chunk|table|image))_(\d+)")


def document_key(path: str) -> str:
    """Short stable key for a document, derived from its filename."""
    name = os.path.basename(path).lower()
    return hashlib.sha1(name.encode()).hexdigest()[:8]


def make_id(doc_key: str, kind: str, index: int) -> str:
    return f"{doc_key}{SEPARATOR}{kind}_{index}"


def parse_id(full_id: str) -> tuple[str | None, str, int]:
    """Split an id into (document key, kind, index).

    Tolerates un-prefixed legacy ids so a store written before namespacing still reads.
    """
    match = ID_PATTERN.fullmatch(full_id)
    if match:
        return match.group(1), match.group(2), int(match.group(3))

    kind, _, number = full_id.rpartition("_")
    return None, kind or full_id, int(number) if number.isdigit() else 0
