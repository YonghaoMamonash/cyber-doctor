import unittest

from rag.rag_chain import optimize_context_with_self_rag


class StubClient:
    def __init__(self, responses):
        self._responses = responses
        self._idx = 0

    def chat_with_ai(self, _prompt):
        if self._idx >= len(self._responses):
            return self._responses[-1]
        value = self._responses[self._idx]
        self._idx += 1
        return value


class RagChainSelfRagTests(unittest.TestCase):
    def test_low_score_triggers_reretrieve(self):
        client = StubClient(
            responses=[
                "初稿答案",
                '{"score":30,"needs_more_retrieval":true,"missing_topics":["并发症"]}',
                "二次初稿",
                '{"score":90,"needs_more_retrieval":false}',
            ]
        )
        contexts = {
            "first": "初始上下文",
            "retry": "重检索上下文",
        }
        state = {"count": 0}

        def fake_retrieve_context(_queries):
            state["count"] += 1
            return contexts["retry"] if state["count"] >= 1 else contexts["first"]

        result = optimize_context_with_self_rag(
            client=client,
            question="糖尿病怎么办",
            rewritten_question="糖尿病怎么办",
            history=[],
            initial_context=contexts["first"],
            retrieve_context_by_queries=fake_retrieve_context,
            self_rag_enabled=True,
            min_grounded_score=70,
            max_retries=1,
            max_seconds=5,
            max_extra_queries=2,
        )
        self.assertEqual(result, contexts["retry"])

    def test_high_score_keeps_initial_context(self):
        client = StubClient(
            responses=[
                "初稿答案",
                '{"score":95,"needs_more_retrieval":false}',
            ]
        )

        def fake_retrieve_context(_queries):
            return "不该被调用"

        result = optimize_context_with_self_rag(
            client=client,
            question="糖尿病怎么办",
            rewritten_question="糖尿病怎么办",
            history=[],
            initial_context="初始上下文",
            retrieve_context_by_queries=fake_retrieve_context,
            self_rag_enabled=True,
            min_grounded_score=70,
            max_retries=1,
            max_seconds=5,
            max_extra_queries=2,
        )
        self.assertEqual(result, "初始上下文")


if __name__ == "__main__":
    unittest.main()
