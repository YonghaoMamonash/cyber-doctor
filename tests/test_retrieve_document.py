import unittest

from rag.retrieve.retrieve_document import retrieve_docs_for_queries


class FakeDoc:
    def __init__(self, page_content, source):
        self.page_content = page_content
        self.metadata = {"source": source}


class RetrieveDocumentTests(unittest.TestCase):
    def test_retrieve_docs_for_queries_merges_and_limits(self):
        mapping = {
            "q1": [FakeDoc("A", "s1"), FakeDoc("B", "s2")],
            "q2": [FakeDoc("A", "s1"), FakeDoc("C", "s3")],
        }

        def fake_retrieve(query):
            return mapping.get(query, [])

        docs, context = retrieve_docs_for_queries(
            queries=["q1", "q2"],
            top_k_per_query=2,
            max_context_docs=3,
            retrieve_fn=fake_retrieve,
        )
        self.assertEqual(len(docs), 3)
        self.assertIn("A", context)
        self.assertIn("B", context)
        self.assertIn("C", context)


if __name__ == "__main__":
    unittest.main()
