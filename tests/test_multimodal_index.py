import os
import tempfile
import unittest

from model.RAG.multimodal_index import build_multimodal_documents


class MultiModalIndexTests(unittest.TestCase):
    def test_build_multimodal_documents_for_csv_and_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "table.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("name,age\nalice,20\nbob,30\n")

            image_path = os.path.join(tmpdir, "xray_scan.png")
            with open(image_path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")

            docs, parent_map = build_multimodal_documents(tmpdir, max_summary_chars=200)

            self.assertGreaterEqual(len(docs), 2)
            modalities = {doc.metadata.get("modality") for doc in docs}
            self.assertIn("table", modalities)
            self.assertIn("image", modalities)
            self.assertTrue(parent_map)

            table_doc = next(doc for doc in docs if doc.metadata.get("modality") == "table")
            self.assertIn("name", table_doc.page_content)
            self.assertIn("alice", table_doc.page_content)


if __name__ == "__main__":
    unittest.main()
