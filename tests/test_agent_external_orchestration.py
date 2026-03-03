import unittest
from unittest.mock import patch

from qa.agent_orchestrator import prepare_agent_inputs
from qa.purpose_type import userPurposeType


def _fake_get_config(path, default):
    mapping = {
        ("model", "agent", "planning", "enabled"): True,
        ("model", "agent", "planning", "llm-decision-enabled"): False,
        ("model", "agent", "planning", "allow-override-explicit-intent"): False,
        ("model", "agent", "memory", "short-term", "enabled"): True,
        ("model", "agent", "memory", "short-term", "max-turns"): 8,
        ("model", "agent", "memory", "short-term", "max-message-chars"): 600,
        ("model", "agent", "memory", "long-term", "enabled"): False,
        ("model", "agent", "memory", "long-term", "top-k"): 3,
        ("model", "agent", "external", "tool-advice", "enabled"): True,
        ("model", "agent", "external", "tool-advice", "max-mcp-suggestions"): 1,
        ("model", "agent", "external", "a2a", "enabled"): True,
        (
            "model",
            "agent",
            "external",
            "mcp",
            "catalog",
        ): [
            {
                "id": "exa",
                "name": "Exa Search",
                "description": "search",
                "tags": ["search", "retrieval"],
            }
        ],
    }
    return mapping.get(tuple(path), default)


class AgentExternalOrchestrationTests(unittest.TestCase):
    @patch("qa.agent_orchestrator._get_config_value", side_effect=_fake_get_config)
    @patch(
        "qa.agent_orchestrator.decide_purpose_and_question",
        return_value=(userPurposeType.RAG, "优化后问题"),
    )
    @patch("qa.agent_orchestrator.prepare_memory_context", return_value=("记忆后问题", []))
    def test_external_advice_injected(
        self, _mock_memory, _mock_plan, _mock_config
    ):
        purpose, question, history = prepare_agent_inputs(
            question="帮我查文献",
            history=[],
            question_type=userPurposeType.text,
        )
        self.assertEqual(purpose, userPurposeType.RAG)
        self.assertEqual(history, [])
        self.assertIn("外部工具建议", question)
        self.assertIn("Exa Search", question)
        self.assertIn("A2A", question)


if __name__ == "__main__":
    unittest.main()
