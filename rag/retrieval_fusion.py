import hashlib
from typing import List


SEPARATOR = "\n-------------分割线--------------\n"


def _doc_identity(doc) -> str:
    source = ""
    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
        source = str(doc.metadata.get("source", ""))
    content = str(getattr(doc, "page_content", ""))
    raw = f"{source}\n{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def deduplicate_and_limit(documents: List, max_docs: int) -> List:
    if max_docs <= 0:
        return []

    unique = []
    seen = set()
    for doc in documents:
        key = _doc_identity(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
        if len(unique) >= max_docs:
            break
    return unique


def build_context(documents: List) -> str:
    return SEPARATOR.join(str(getattr(doc, "page_content", "")) for doc in documents)
