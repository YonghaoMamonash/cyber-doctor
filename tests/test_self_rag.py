import unittest

from rag.self_rag import (
    build_refined_queries,
    parse_self_rag_eval,
    should_retry_retrieval,
)


class SelfRagTests(unittest.TestCase):
    def test_parse_self_rag_eval_json(self):
        result = parse_self_rag_eval(
            '{"score":40,"needs_more_retrieval":true,"missing_topics":["并发症"]}'
        )
        self.assertEqual(result["score"], 40)
        self.assertTrue(result["needs_more_retrieval"])
        self.assertEqual(result["missing_topics"], ["并发症"])

    def test_parse_self_rag_eval_text_fallback(self):
        result = parse_self_rag_eval("score: 75; reason: context is weak")
        self.assertEqual(result["score"], 75)

    def test_should_retry_retrieval(self):
        self.assertTrue(
            should_retry_retrieval(
                {"score": 30, "needs_more_retrieval": False},
                min_grounded_score=70,
            )
        )
        self.assertTrue(
            should_retry_retrieval(
                {"score": 90, "needs_more_retrieval": True},
                min_grounded_score=70,
            )
        )
        self.assertFalse(
            should_retry_retrieval(
                {"score": 90, "needs_more_retrieval": False},
                min_grounded_score=70,
            )
        )

    def test_build_refined_queries(self):
        queries = build_refined_queries(
            base_question="糖尿病如何治疗",
            eval_result={
                "missing_topics": ["并发症", "饮食"],
                "reason": "缺乏并发症和饮食建议",
            },
            max_extra_queries=2,
        )
        self.assertEqual(queries[0], "糖尿病如何治疗")
        self.assertEqual(len(queries), 3)
        self.assertIn("并发症", queries[1])


if __name__ == "__main__":
    unittest.main()
