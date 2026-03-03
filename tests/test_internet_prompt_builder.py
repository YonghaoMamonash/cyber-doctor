import unittest

from Internet.Internet_chain import build_internet_prompt


class InternetPromptBuilderTests(unittest.TestCase):
    def test_prefers_html_context(self):
        prompt = build_internet_prompt(
            question="q",
            html_context="HTML上下文",
            snippet_context="摘要上下文",
        )
        self.assertIn("HTML上下文", prompt)
        self.assertNotIn("摘要上下文", prompt)

    def test_fallback_to_snippet_context(self):
        prompt = build_internet_prompt(
            question="q",
            html_context="",
            snippet_context="摘要上下文",
        )
        self.assertIn("摘要上下文", prompt)

    def test_no_context_returns_question(self):
        prompt = build_internet_prompt(
            question="q",
            html_context="",
            snippet_context="",
        )
        self.assertEqual(prompt, "q")


if __name__ == "__main__":
    unittest.main()
