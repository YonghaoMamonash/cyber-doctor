import unittest

from qa.agent_planner import decide_purpose_and_question, parse_react_plan
from qa.purpose_type import userPurposeType


class StubClient:
    def __init__(self, response: str):
        self._response = response

    def chat_using_messages(self, _messages):
        return self._response


class AgentPlannerTests(unittest.TestCase):
    def test_parse_react_plan_valid_json(self):
        action, action_input = parse_react_plan(
            '{"action":"InternetSearch","action_input":"最新流感防治指南"}'
        )
        self.assertEqual(action, "InternetSearch")
        self.assertEqual(action_input, "最新流感防治指南")

    def test_parse_react_plan_invalid_returns_none(self):
        action, action_input = parse_react_plan("not-json")
        self.assertIsNone(action)
        self.assertIsNone(action_input)

    def test_decide_purpose_uses_llm_action_for_text_query(self):
        client = StubClient(
            '{"action":"RAG","action_input":"根据知识库解释糖尿病饮食"}'
        )
        purpose, rewritten = decide_purpose_and_question(
            question="糖尿病饮食建议",
            history=[],
            original_purpose=userPurposeType.text,
            llm_client=client,
            planning_enabled=True,
            llm_decision_enabled=True,
        )
        self.assertEqual(purpose, userPurposeType.RAG)
        self.assertEqual(rewritten, "根据知识库解释糖尿病饮食")

    def test_decide_purpose_keeps_explicit_intent(self):
        client = StubClient('{"action":"text","action_input":"随便改写"}')
        purpose, rewritten = decide_purpose_and_question(
            question="帮我搜索流感最新消息",
            history=[],
            original_purpose=userPurposeType.InternetSearch,
            llm_client=client,
            planning_enabled=True,
            llm_decision_enabled=True,
        )
        self.assertEqual(purpose, userPurposeType.InternetSearch)
        self.assertEqual(rewritten, "帮我搜索流感最新消息")


if __name__ == "__main__":
    unittest.main()
