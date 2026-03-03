import unittest
from unittest.mock import Mock, patch

from qa.answer import get_answer
from qa.purpose_type import userPurposeType


class AnswerAgentOrchestrationTests(unittest.TestCase):
    @patch("qa.answer.map_question_to_function")
    @patch("qa.answer.prepare_agent_inputs")
    def test_get_answer_uses_orchestrated_inputs(self, mock_prepare, mock_map):
        prepared_history = [["u", "a"]]
        mock_prepare.return_value = (
            userPurposeType.RAG,
            "优化后的问题",
            prepared_history,
        )
        mock_function = Mock(return_value=("STREAM", userPurposeType.RAG))
        mock_map.return_value = mock_function

        result = get_answer(
            question="原问题",
            history=[["old_u", "old_a"]],
            question_type=userPurposeType.text,
            image_url=None,
        )

        mock_map.assert_called_once_with(userPurposeType.RAG)
        mock_function.assert_called_once_with(
            userPurposeType.RAG,
            "优化后的问题",
            prepared_history,
            None,
        )
        self.assertEqual(result, ("STREAM", userPurposeType.RAG))


if __name__ == "__main__":
    unittest.main()
