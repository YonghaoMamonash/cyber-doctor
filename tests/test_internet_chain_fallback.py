import unittest
from unittest.mock import patch

from Internet.Internet_chain import InternetSearchChain


class InternetChainFallbackTests(unittest.TestCase):
    @patch("Internet.Internet_chain.retrieve_html", side_effect=RuntimeError("boom"))
    @patch("Internet.Internet_chain.has_html_files", return_value=True)
    @patch("Internet.Internet_chain.search_baidu", return_value=None)
    @patch("Internet.Internet_chain.search_bing", return_value=None)
    @patch("Internet.Internet_chain.extract_question", return_value="子问题1;子问题2")
    def test_fallback_to_plain_prompt_when_retrieve_fails(
        self,
        _extract_question,
        _search_bing,
        _search_baidu,
        _has_html_files,
        _retrieve_html,
    ):
        with patch("Internet.Internet_chain.Clientfactory") as mock_factory:
            mock_client = mock_factory.return_value.get_client.return_value
            mock_client.chat_with_ai_stream.return_value = "STREAM"

            response, links, success = InternetSearchChain("原问题", history=[])

            self.assertEqual(response, "STREAM")
            self.assertEqual(links, {})
            self.assertFalse(success)

    @patch("Internet.Internet_chain.retrieve_html", side_effect=RuntimeError("boom"))
    @patch("Internet.Internet_chain.has_html_files", return_value=True)
    @patch("Internet.Internet_chain.extract_question", return_value="子问题1")
    def test_snippet_fallback_marks_success_when_available(
        self,
        _extract_question,
        _has_html_files,
        _retrieve_html,
    ):
        def fake_search(_query, _links, _links_lock, hits, _hits_lock, _num_results=3):
            hits.append(
                {
                    "title": "养生知识",
                    "link": "https://example.com",
                    "snippet": "养生建议",
                }
            )

        with patch("Internet.Internet_chain.search_bing", side_effect=fake_search):
            with patch("Internet.Internet_chain.search_baidu", side_effect=fake_search):
                with patch("Internet.Internet_chain.Clientfactory") as mock_factory:
                    mock_client = mock_factory.return_value.get_client.return_value
                    mock_client.chat_with_ai_stream.return_value = "STREAM"

                    response, links, success = InternetSearchChain("原问题", history=[])

                    self.assertEqual(response, "STREAM")
                    self.assertTrue(success)
                    self.assertIn("https://example.com", links)


if __name__ == "__main__":
    unittest.main()
