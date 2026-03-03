import unittest

from rag.query_optimizer import parse_queries, generate_queries


class StubClient:
    def __init__(self, response: str):
        self._response = response

    def chat_using_messages(self, _messages):
        return self._response


class QueryOptimizerTests(unittest.TestCase):
    def test_parse_queries_from_json_array(self):
        raw = '["糖尿病饮食建议","糖尿病运动建议","糖尿病饮食建议"]'
        parsed = parse_queries(raw, fallback="糖尿病怎么管理")
        self.assertEqual(parsed, ["糖尿病饮食建议", "糖尿病运动建议"])

    def test_parse_queries_from_lines(self):
        raw = "1. 糖尿病饮食建议\n2) 糖尿病运动建议\n糖尿病饮食建议"
        parsed = parse_queries(raw, fallback="糖尿病怎么管理")
        self.assertEqual(parsed, ["糖尿病饮食建议", "糖尿病运动建议"])

    def test_parse_queries_fallback_to_original_when_empty(self):
        parsed = parse_queries("", fallback="原问题")
        self.assertEqual(parsed, ["原问题"])

    def test_generate_queries_fallback_when_model_output_invalid(self):
        client = StubClient("???")
        queries = generate_queries(
            question="原问题",
            history=[],
            count=3,
            llm_client=client,
        )
        self.assertEqual(queries, ["原问题"])


if __name__ == "__main__":
    unittest.main()
