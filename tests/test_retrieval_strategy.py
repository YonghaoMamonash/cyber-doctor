import unittest

from rag.retrieval_strategy import (
    choose_multi_query_count,
    choose_rewrite_enabled,
    estimate_query_complexity,
    is_followup_question,
)


class RetrievalStrategyTests(unittest.TestCase):
    def test_is_followup_question_true_for_short_followup(self):
        history = [["糖尿病症状有哪些", "常见症状是..."]]
        self.assertTrue(is_followup_question("那怎么治疗", history))

    def test_choose_rewrite_enabled_followup_only(self):
        history = [["A", "B"]]
        self.assertTrue(
            choose_rewrite_enabled(
                rewrite_enabled=True,
                rewrite_mode="followup-only",
                question="那有什么副作用",
                history=history,
            )
        )
        self.assertFalse(
            choose_rewrite_enabled(
                rewrite_enabled=True,
                rewrite_mode="followup-only",
                question="糖尿病是什么",
                history=[],
            )
        )

    def test_estimate_query_complexity(self):
        self.assertEqual(estimate_query_complexity("什么是糖尿病"), 0)
        self.assertGreaterEqual(
            estimate_query_complexity("比较糖尿病和高血压的治疗方案，并给出副作用"),
            2,
        )

    def test_choose_multi_query_count(self):
        self.assertEqual(
            choose_multi_query_count(
                question="什么是糖尿病",
                history=[],
                configured_mode="auto",
                multi_query_enabled=True,
                multi_query_count=3,
            ),
            1,
        )
        self.assertEqual(
            choose_multi_query_count(
                question="比较糖尿病和高血压的治疗方案，并给出副作用",
                history=[],
                configured_mode="auto",
                multi_query_enabled=True,
                multi_query_count=3,
            ),
            3,
        )
        self.assertEqual(
            choose_multi_query_count(
                question="随便问问",
                history=[],
                configured_mode="single",
                multi_query_enabled=True,
                multi_query_count=3,
            ),
            1,
        )
        self.assertEqual(
            choose_multi_query_count(
                question="随便问问",
                history=[],
                configured_mode="multi",
                multi_query_enabled=True,
                multi_query_count=4,
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
