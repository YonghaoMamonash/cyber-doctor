import unittest

from Internet.search_utils import (
    build_snippet_context,
    choose_effective_search_question,
    extract_real_url,
    rank_hits_by_query,
)


class SearchUtilsTests(unittest.TestCase):
    def test_extract_real_url_from_bing_redirect(self):
        url = (
            "https://www.bing.com/ck/a?!&&p=xxx"
            "&u=a1aHR0cHM6Ly93d3cuZXhhbXBsZS5jb20vcGFnZQ&ntb=1"
        )
        self.assertEqual(extract_real_url(url), "https://www.example.com/page")

    def test_extract_real_url_keep_normal_link(self):
        url = "https://www.example.com/x"
        self.assertEqual(extract_real_url(url), url)

    def test_rank_hits_deduplicates_and_orders(self):
        hits = [
            {"title": "养生知识", "link": "https://a.com", "snippet": "养生方法"},
            {"title": "无关", "link": "https://b.com", "snippet": "abc"},
            {"title": "重复", "link": "https://a.com", "snippet": "重复项"},
        ]
        ranked = rank_hits_by_query(hits, query="养生知识", max_items=5)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["link"], "https://a.com")

    def test_build_snippet_context_not_empty(self):
        hits = [{"title": "标题", "link": "https://a.com", "snippet": "摘要"}]
        context = build_snippet_context(hits, max_items=3)
        self.assertIn("标题", context)
        self.assertIn("https://a.com", context)
        self.assertIn("摘要", context)

    def test_choose_effective_search_question_fallback_to_original(self):
        original = "帮我搜索一下养生知识"
        extracted = "笔记本电脑连接wifi提示无法连接到网络怎么办"
        self.assertEqual(
            choose_effective_search_question(original, extracted),
            original,
        )

    def test_choose_effective_search_question_use_extracted_when_related(self):
        original = "帮我搜索一下养生知识"
        extracted = "养生小知识有哪些"
        self.assertEqual(
            choose_effective_search_question(original, extracted),
            extracted,
        )


if __name__ == "__main__":
    unittest.main()
