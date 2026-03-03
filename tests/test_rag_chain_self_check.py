import unittest

from rag.rag_chain import (
    build_self_check_prompt,
    generate_response_with_optional_self_check,
)


class StubClient:
    def __init__(self, draft: str):
        self.draft = draft
        self.last_stream_prompt = None
        self.stream_calls = 0
        self.non_stream_calls = 0

    def chat_with_ai(self, prompt):
        self.non_stream_calls += 1
        return self.draft

    def chat_with_ai_stream(self, prompt, _history):
        self.stream_calls += 1
        self.last_stream_prompt = prompt
        return "STREAM"


class RagChainSelfCheckTests(unittest.TestCase):
    def test_build_self_check_prompt_contains_all_parts(self):
        prompt = build_self_check_prompt(
            context="文档上下文",
            question="用户问题",
            draft_answer="初稿答案",
        )
        self.assertIn("文档上下文", prompt)
        self.assertIn("用户问题", prompt)
        self.assertIn("初稿答案", prompt)

    def test_self_check_enabled_uses_review_prompt(self):
        client = StubClient(draft="初稿答案")
        result = generate_response_with_optional_self_check(
            client=client,
            prompt="基础提示词",
            context="文档上下文",
            question="用户问题",
            history=[],
            self_check_enabled=True,
        )
        self.assertEqual(result, "STREAM")
        self.assertEqual(client.non_stream_calls, 1)
        self.assertEqual(client.stream_calls, 1)
        self.assertIn("初稿答案", client.last_stream_prompt)

    def test_self_check_disabled_streams_original_prompt(self):
        client = StubClient(draft="初稿答案")
        result = generate_response_with_optional_self_check(
            client=client,
            prompt="基础提示词",
            context="文档上下文",
            question="用户问题",
            history=[],
            self_check_enabled=False,
        )
        self.assertEqual(result, "STREAM")
        self.assertEqual(client.non_stream_calls, 0)
        self.assertEqual(client.last_stream_prompt, "基础提示词")


if __name__ == "__main__":
    unittest.main()
