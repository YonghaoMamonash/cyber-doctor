import json
from typing import List

from langchain_core.documents import Document


def build_vector_query_payload(
    index_name: str,
    embedding: List[float],
    top_k: int,
    text_property: str = "text",
    metadata_property: str = "metadata_json",
):
    cypher = (
        "CALL db.index.vector.queryNodes($index_name, $k, $embedding) "
        "YIELD node, score "
        f"RETURN node.{text_property} AS text, "
        f"node.{metadata_property} AS metadata_json, score"
    )
    return {
        "cypher": cypher,
        "params": {
            "index_name": index_name,
            "k": int(top_k),
            "embedding": embedding,
        },
    }


def _safe_json_dumps(data) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return "{}"


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return {}


class Neo4jVectorRetriever:
    def __init__(
        self,
        graph,
        embedding_model,
        index_name: str,
        top_k: int,
        text_property: str = "text",
        metadata_property: str = "metadata_json",
    ):
        self._graph = graph
        self._embedding_model = embedding_model
        self._index_name = index_name
        self._top_k = top_k
        self._text_property = text_property
        self._metadata_property = metadata_property

    def invoke(self, query: str) -> List[Document]:
        embedding = self._embedding_model.embed_query(query)
        payload = build_vector_query_payload(
            index_name=self._index_name,
            embedding=embedding,
            top_k=self._top_k,
            text_property=self._text_property,
            metadata_property=self._metadata_property,
        )
        rows = self._graph.run(payload["cypher"], **payload["params"]).data()
        docs = []
        for row in rows:
            text = str(row.get("text", "") or "")
            metadata = _safe_json_loads(str(row.get("metadata_json", "{}")))
            if not isinstance(metadata, dict):
                metadata = {}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs


def build_neo4j_vector_retriever(
    graph,
    embedding_model,
    docs: List[Document],
    index_name: str,
    label: str,
    top_k: int,
    embedding_property: str = "embedding",
    text_property: str = "text",
    metadata_property: str = "metadata_json",
    reset_before_build: bool = True,
):
    if not docs:
        return None

    texts = [doc.page_content for doc in docs]
    embeddings = embedding_model.embed_documents(texts)
    if not embeddings:
        return None

    if reset_before_build:
        graph.run(f"MATCH (n:{label}) WHERE n.__source='rag' DETACH DELETE n")

    rows = []
    for doc, emb in zip(docs, embeddings):
        rows.append(
            {
                "text": doc.page_content,
                "embedding": emb,
                "metadata_json": _safe_json_dumps(doc.metadata or {}),
                "source_doc_id": str((doc.metadata or {}).get("source_doc_id", "")),
            }
        )

    upsert_query = (
        "UNWIND $rows AS row "
        f"CREATE (n:{label}) "
        f"SET n.{text_property}=row.text, "
        f"n.{embedding_property}=row.embedding, "
        f"n.{metadata_property}=row.metadata_json, "
        "n.source_doc_id=row.source_doc_id, "
        "n.__source='rag'"
    )
    graph.run(upsert_query, rows=rows)

    dim = len(embeddings[0])
    create_index_query = (
        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
        f"FOR (n:{label}) ON (n.{embedding_property}) "
        "OPTIONS {indexConfig: {"
        f"`vector.dimensions`: {dim}, "
        "`vector.similarity_function`: 'cosine'}}"
    )
    graph.run(create_index_query)

    return Neo4jVectorRetriever(
        graph=graph,
        embedding_model=embedding_model,
        index_name=index_name,
        top_k=top_k,
        text_property=text_property,
        metadata_property=metadata_property,
    )
