import unittest
from unittest.mock import patch

from qa.agent_orchestrator import prepare_agent_inputs
from qa.purpose_type import userPurposeType


def _fake_get_config(path, default):
    mapping = {
        ("model", "agent", "planning", "enabled"): True,
        ("model", "agent", "planning", "llm-decision-enabled"): False,
        ("model", "agent", "planning", "allow-override-explicit-intent"): False,
        ("model", "agent", "memory", "short-term", "enabled"): False,
        ("model", "agent", "memory", "short-term", "max-turns"): 8,
        ("model", "agent", "memory", "short-term", "max-message-chars"): 600,
        ("model", "agent", "memory", "long-term", "enabled"): False,
        ("model", "agent", "memory", "long-term", "top-k"): 3,
        ("model", "agent", "external", "tool-advice", "enabled"): False,
        ("model", "agent", "external", "a2a", "enabled"): True,
        ("model", "agent", "external", "a2a", "mode"): "assist",
        ("model", "agent", "external", "a2a", "min-question-chars"): 6,
        ("model", "agent", "external", "a2a", "timeout-seconds"): 10,
        ("model", "agent", "external", "a2a", "endpoint"): "http://a2a.local/rpc",
    }
    return mapping.get(tuple(path), default)


class AgentA2ADelegationTests(unittest.TestCase):
    @patch("qa.agent_orchestrator._get_config_value", side_effect=_fake_get_config)
    @patch(
        "qa.agent_orchestrator.decide_purpose_and_question",
        return_value=(userPurposeType.RAG, "优化后问题"),
    )
    @patch("qa.agent_orchestrator.prepare_memory_context", return_value=("记忆后问题", []))
    @patch("qa.agent_orchestrator.delegate_to_a2a", return_value="外部Agent给出的补充结论")
    def test_a2a_assist_injected(
        self, _mock_delegate, _mock_memory, _mock_plan, _mock_config
    ):
        purpose, question, _history = prepare_agent_inputs(
            question="请你综合分析慢性病管理方案",
            history=[],
            question_type=userPurposeType.text,
        )
        self.assertEqual(purpose, userPurposeType.RAG)
        self.assertIn("外部Agent补充信息", question)
        self.assertIn("外部Agent给出的补充结论", question)


if __name__ == "__main__":
    unittest.main()
