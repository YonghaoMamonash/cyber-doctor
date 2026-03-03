from typing import Callable, List, Tuple
from langchain_core.documents import Document

from rag.retrieval_fusion import build_context, deduplicate_and_limit
from utils.console import safe_print


def _default_retrieve(query: str) -> List[Document]:
    from model.RAG.retrieve_service import retrieve

    return retrieve(query)


def format_docs(docs: List[Document]):
    return build_context(docs)


def retrieve_docs_for_queries(
    queries: List[str],
    top_k_per_query: int = 6,
    max_context_docs: int = 8,
    retrieve_fn: Callable[[str], List[Document]] | None = None,
) -> Tuple[List[Document], str]:
    retriever = retrieve_fn or _default_retrieve
    collected: List[Document] = []

    for query in queries:
        docs = retriever(query) or []
        if top_k_per_query > 0:
            docs = docs[:top_k_per_query]
        collected.extend(docs)

    merged_docs = deduplicate_and_limit(collected, max_docs=max_context_docs)
    context = format_docs(merged_docs)
    return merged_docs, context


def retrieve_docs(question: str) -> Tuple[List[Document], str]:
    docs, _context = retrieve_docs_for_queries(queries=[question])
    safe_print(_context[:800])
    return docs, _context
    
