import unittest

from qa.kg_relation_filter import normalize_relationships


class KgRelationFilterTests(unittest.TestCase):
    def test_filter_and_deduplicate_relationships(self):
        rows = [
            ("糖尿病", "宜吃", "燕麦"),
            ("糖尿病", "症状", "多饮"),
            ("糖尿病", "宜吃", "燕麦"),
        ]
        result = normalize_relationships(rows, allowed_types={"宜吃"}, max_items=10)
        self.assertEqual(result, ["糖尿病 宜吃 燕麦"])

    def test_limit_relationships(self):
        rows = [
            ("A", "r1", "B"),
            ("A", "r2", "C"),
            ("A", "r3", "D"),
        ]
        result = normalize_relationships(rows, allowed_types=None, max_items=2)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
