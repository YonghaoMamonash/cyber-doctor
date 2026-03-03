import unittest

from model.RAG.neo4j_vector_bridge import build_vector_query_payload


class Neo4jVectorBridgeTests(unittest.TestCase):
    def test_build_vector_query_payload(self):
        payload = build_vector_query_payload(
            index_name="rag_chunks",
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
        )
        self.assertIn("db.index.vector.queryNodes", payload["cypher"])
        self.assertEqual(payload["params"]["index_name"], "rag_chunks")
        self.assertEqual(payload["params"]["k"], 5)
        self.assertEqual(payload["params"]["embedding"], [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
