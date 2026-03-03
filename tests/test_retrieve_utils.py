import unittest

from model.RAG.retrieve_utils import attach_last_modified_metadata, filter_stale_documents


class FakeDoc:
    def __init__(self, source):
        self.page_content = "x"
        self.metadata = {"source": source}


class RetrieveUtilsTests(unittest.TestCase):
    def test_attach_last_modified_metadata(self):
        docs = [FakeDoc("a.txt"), FakeDoc("b.txt")]
        mapping = {"a.txt": 100.0, "b.txt": 200.0}

        def fake_getmtime(path):
            return mapping[path]

        updated = attach_last_modified_metadata(docs, getmtime_fn=fake_getmtime)
        self.assertEqual(updated[0].metadata["last_modified_ts"], 100.0)
        self.assertEqual(updated[1].metadata["last_modified_ts"], 200.0)

    def test_filter_stale_documents(self):
        docs = [FakeDoc("new.txt"), FakeDoc("old.txt")]
        docs[0].metadata["last_modified_ts"] = 1_700_000_000
        docs[1].metadata["last_modified_ts"] = 1_600_000_000

        kept = filter_stale_documents(
            docs,
            stale_days=30,
            now_ts=1_700_000_000,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].metadata["source"], "new.txt")


if __name__ == "__main__":
    unittest.main()
