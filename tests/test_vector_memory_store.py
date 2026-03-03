import os
import tempfile
import unittest

from qa.vector_memory_store import PersistentVectorMemoryStore


class VectorMemoryStoreTests(unittest.TestCase):
    def test_add_and_search_persistent_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "memory.jsonl")
            store = PersistentVectorMemoryStore(
                file_path=file_path,
                max_records=100,
                vector_dim=64,
            )
            store.add_facts("s1", ["过敏: 青霉素", "偏好: 清淡饮食"])
            store.add_facts("s2", ["疾病史: 高血压"])

            result = store.search_facts("s1", "我能用青霉素吗", top_k=2)
            self.assertTrue(result)
            self.assertIn("青霉素", result[0])

            self.assertTrue(os.path.exists(file_path))
            reload_store = PersistentVectorMemoryStore(
                file_path=file_path,
                max_records=100,
                vector_dim=64,
            )
            reload_result = reload_store.search_facts("s1", "饮食建议", top_k=2)
            self.assertTrue(any("清淡饮食" in x for x in reload_result))


if __name__ == "__main__":
    unittest.main()
