import unittest

from rag.rag_chain import build_rag_prompt


class RagChainPromptTests(unittest.TestCase):
    def test_build_prompt_without_context(self):
        prompt = build_rag_prompt(context="", question="什么是糖尿病？")
        self.assertEqual(prompt, "什么是糖尿病？")

    def test_build_prompt_with_context(self):
        prompt = build_rag_prompt(context="文档片段A", question="什么是糖尿病？")
        self.assertIn("文档片段A", prompt)
        self.assertIn("什么是糖尿病？", prompt)


if __name__ == "__main__":
    unittest.main()
