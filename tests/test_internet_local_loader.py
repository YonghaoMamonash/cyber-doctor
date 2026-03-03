import os
import tempfile
import unittest

from model.Internet.local_loader import load_local_html_documents


class InternetLocalLoaderTests(unittest.TestCase):
    def test_load_local_html_documents_extracts_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "a.html")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("<html><body><h1>标题</h1><p>正文内容</p></body></html>")

            docs = load_local_html_documents(tmpdir)
            self.assertEqual(len(docs), 1)
            self.assertIn("标题", docs[0].page_content)
            self.assertIn("正文内容", docs[0].page_content)
            self.assertEqual(docs[0].metadata.get("source"), fpath)


if __name__ == "__main__":
    unittest.main()
