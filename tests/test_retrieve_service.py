import unittest

from model.RAG.retrieve_service import retrieve, retrieve_with_raptor_lite


class StubRetriever:
    def __init__(self, value):
        self._value = value

    def invoke(self, _query):
        return self._value


class StubInstance:
    def __init__(self, user_id, base_docs, user_docs=None):
        self.user_id = user_id
        self.retriever = StubRetriever(base_docs)
        self._user_retriever = StubRetriever(user_docs) if user_docs is not None else None

    def get_user_retriever(self):
        return self._user_retriever


class FakeDoc:
    def __init__(self, source_doc_id):
        self.page_content = "summary"
        self.metadata = {"source_doc_id": source_doc_id}


class StubSummaryInstance(StubInstance):
    def __init__(self):
        super().__init__(user_id=None, base_docs=["base"], user_docs=None)
        self.summary_retriever = StubRetriever([FakeDoc("sid-1"), FakeDoc("sid-2")])
        self._map = {
            "sid-1": ["chunk-1a", "chunk-1b"],
            "sid-2": ["chunk-2a"],
        }

    def retrieve_chunks_by_source_ids(self, source_ids, limit=6):
        merged = []
        for sid in source_ids:
            merged.extend(self._map.get(sid, []))
        return merged[:limit]


class RetrieveServiceTests(unittest.TestCase):
    def test_fallback_to_base_retriever_when_user_retriever_missing(self):
        instance = StubInstance(user_id="u1", base_docs=["base"], user_docs=None)
        docs = retrieve("q", instance=instance)
        self.assertEqual(docs, ["base"])

    def test_use_user_retriever_when_available(self):
        instance = StubInstance(user_id="u1", base_docs=["base"], user_docs=["user"])
        docs = retrieve("q", instance=instance)
        self.assertEqual(docs, ["user"])

    def test_retrieve_with_raptor_lite_prefers_summary_mapping(self):
        instance = StubSummaryInstance()
        docs = retrieve_with_raptor_lite(
            "q",
            instance=instance,
            summary_top_k=2,
            chunk_top_k=2,
        )
        self.assertEqual(docs, ["chunk-1a", "chunk-1b"])

    def test_retrieve_with_raptor_lite_fallback_to_base(self):
        instance = StubInstance(user_id=None, base_docs=["base"], user_docs=None)
        docs = retrieve_with_raptor_lite("q", instance=instance)
        self.assertEqual(docs, ["base"])


if __name__ == "__main__":
    unittest.main()
