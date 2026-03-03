import hashlib
import re
from collections import OrderedDict
from typing import Dict, List, Tuple

from langchain_core.documents import Document


_SPACE_PATTERN = re.compile(r"\s+")
_SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?；;\n]")


def _clean_text(text: str) -> str:
    return _SPACE_PATTERN.sub(" ", (text or "")).strip()


def summarize_text(text: str, max_chars: int = 260) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    if max_chars <= 0:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned

    sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(cleaned) if s.strip()]
    if not sentences:
        return cleaned[:max_chars]

    merged = []
    current_len = 0
    for sentence in sentences:
        if current_len + len(sentence) > max_chars:
            break
        merged.append(sentence)
        current_len += len(sentence)

    if merged:
        return "".join(merged)
    return cleaned[:max_chars]


def _build_source_doc_id(source: str, content: str) -> str:
    if source:
        payload = source
    else:
        payload = content[:300]
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _extract_source(doc: Document) -> str:
    if not isinstance(doc.metadata, dict):
        return ""
    return str(doc.metadata.get("source", ""))


def build_summary_layer(
    chunks: List[Document],
    max_summary_chars: int = 260,
) -> Tuple[List[Document], Dict[str, List[Document]]]:
    grouped = OrderedDict()
    for chunk in chunks:
        source = _extract_source(chunk)
        source_doc_id = (
            str(chunk.metadata.get("source_doc_id", ""))
            if isinstance(chunk.metadata, dict)
            else ""
        )
        if not source_doc_id:
            source_doc_id = _build_source_doc_id(source=source, content=chunk.page_content)
        grouped.setdefault(source_doc_id, []).append(chunk)

    summary_docs: List[Document] = []
    source_map: Dict[str, List[Document]] = {}

    for source_doc_id, doc_chunks in grouped.items():
        source = _extract_source(doc_chunks[0]) if doc_chunks else ""
        source_map[source_doc_id] = doc_chunks

        merged_text = " ".join(_clean_text(x.page_content) for x in doc_chunks[:8])
        summary_text = summarize_text(merged_text, max_chars=max_summary_chars)
        if not summary_text:
            continue

        summary_docs.append(
            Document(
                page_content=summary_text,
                metadata={
                    "source_doc_id": source_doc_id,
                    "source": source,
                    "is_summary": True,
                    "chunk_count": len(doc_chunks),
                },
            )
        )

    return summary_docs, source_map
