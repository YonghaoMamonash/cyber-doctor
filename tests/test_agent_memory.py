import unittest

from qa.agent_memory import (
    SessionMemoryStore,
    compress_history,
    enrich_question_with_memory,
    extract_user_facts,
)


class AgentMemoryTests(unittest.TestCase):
    def test_compress_history_keeps_recent_turns(self):
        history = [
            ["用户第一轮提问", "助手第一轮回答"],
            ["用户第二轮提问", "助手第二轮回答"],
            ["用户第三轮提问", "助手第三轮回答"],
        ]
        compressed = compress_history(history, keep_recent_turns=2, max_message_chars=8)
        self.assertEqual(len(compressed), 2)
        self.assertTrue(all(len(turn[0]) <= 8 for turn in compressed))

    def test_extract_user_facts(self):
        facts = extract_user_facts("我叫张三，我对青霉素过敏，我有高血压，我喜欢低盐饮食。")
        self.assertIn("姓名: 张三", facts)
        self.assertIn("过敏: 青霉素", facts)
        self.assertIn("疾病史: 高血压", facts)
        self.assertIn("偏好: 低盐饮食", facts)

    def test_memory_store_search(self):
        store = SessionMemoryStore(max_facts_per_session=5)
        store.add_facts("session-1", ["过敏: 青霉素", "疾病史: 高血压"])

        result = store.search_facts("session-1", "我过敏了能吃什么药", top_k=1)
        self.assertEqual(result, ["过敏: 青霉素"])

    def test_enrich_question_with_memory(self):
        store = SessionMemoryStore(max_facts_per_session=5)
        store.add_facts("session-1", ["过敏: 青霉素"])

        enriched = enrich_question_with_memory(
            question="我感冒了该吃什么药",
            session_id="session-1",
            store=store,
            top_k=2,
        )
        self.assertIn("历史记忆", enriched)
        self.assertIn("过敏: 青霉素", enriched)


if __name__ == "__main__":
    unittest.main()
