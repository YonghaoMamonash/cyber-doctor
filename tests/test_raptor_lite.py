import unittest

from model.RAG.raptor_lite import build_summary_layer, summarize_text


class FakeDoc:
    def __init__(self, page_content, source):
        self.page_content = page_content
        self.metadata = {"source": source}


class RaptorLiteTests(unittest.TestCase):
    def test_summarize_text_limits_length(self):
        text = "第一句。第二句。第三句。第四句。"
        summary = summarize_text(text, max_chars=6)
        self.assertTrue(summary)
        self.assertLessEqual(len(summary), 6)

    def test_build_summary_layer_groups_by_source(self):
        chunks = [
            FakeDoc("糖尿病是一种慢性病。", "a.md"),
            FakeDoc("饮食控制很重要。", "a.md"),
            FakeDoc("高血压需要长期管理。", "b.md"),
        ]

        summary_docs, source_map = build_summary_layer(chunks, max_summary_chars=50)

        self.assertEqual(len(summary_docs), 2)
        for doc in summary_docs:
            self.assertTrue(doc.metadata.get("is_summary"))
            self.assertIn("source_doc_id", doc.metadata)

        self.assertEqual(len(source_map), 2)
        total_chunks = sum(len(v) for v in source_map.values())
        self.assertEqual(total_chunks, 3)


if __name__ == "__main__":
    unittest.main()
