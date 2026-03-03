import unittest

from rag.retrieval_fusion import deduplicate_and_limit, build_context


class FakeDoc:
    def __init__(self, page_content, source):
        self.page_content = page_content
        self.metadata = {"source": source}


class RetrievalFusionTests(unittest.TestCase):
    def test_deduplicate_and_limit(self):
        docs = [
            FakeDoc("A", "file1"),
            FakeDoc("A", "file1"),
            FakeDoc("B", "file1"),
            FakeDoc("A", "file2"),
        ]
        merged = deduplicate_and_limit(docs, max_docs=2)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].page_content, "A")
        self.assertEqual(merged[1].page_content, "B")

    def test_build_context_with_separator(self):
        docs = [FakeDoc("第一段", "f1"), FakeDoc("第二段", "f2")]
        context = build_context(docs)
        self.assertIn("第一段", context)
        self.assertIn("第二段", context)
        self.assertIn("-------------分割线--------------", context)


if __name__ == "__main__":
    unittest.main()
