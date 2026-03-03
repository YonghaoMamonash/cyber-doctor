# 该函数用于对外界提供retreive服务，调用的是retrieve_model 中的接口
from typing import List
from langchain_core.documents import Document


def _get_default_instance():
    from model.RAG.retrieve_model import INSTANCE

    return INSTANCE


def retrieve(query: str, instance=None) -> List[Document]:
    active_instance = instance or _get_default_instance()

    if active_instance.user_id is not None:
        user_retriever = active_instance.get_user_retriever()
        if user_retriever is not None:
            return user_retriever.invoke(query)

    return active_instance.retriever.invoke(query)


def retrieve_with_raptor_lite(
    query: str,
    instance=None,
    summary_top_k: int = 3,
    chunk_top_k: int = 6,
) -> List[Document]:
    active_instance = instance or _get_default_instance()

    if active_instance.user_id is not None:
        user_retriever = active_instance.get_user_retriever()
        if user_retriever is not None:
            return user_retriever.invoke(query)

    summary_retriever = getattr(active_instance, "summary_retriever", None)
    if summary_retriever is None:
        return retrieve(query, instance=active_instance)

    try:
        summary_docs = summary_retriever.invoke(query) or []
    except Exception:
        return retrieve(query, instance=active_instance)

    if summary_top_k > 0:
        summary_docs = summary_docs[:summary_top_k]

    source_ids = []
    for doc in summary_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source_doc_id = metadata.get("source_doc_id")
        if source_doc_id and source_doc_id not in source_ids:
            source_ids.append(source_doc_id)

    if not source_ids:
        return retrieve(query, instance=active_instance)

    if not hasattr(active_instance, "retrieve_chunks_by_source_ids"):
        return retrieve(query, instance=active_instance)

    try:
        chunk_docs = active_instance.retrieve_chunks_by_source_ids(
            source_ids=source_ids,
            limit=chunk_top_k,
        )
        if chunk_docs:
            return chunk_docs
    except Exception:
        pass

    return retrieve(query, instance=active_instance)
