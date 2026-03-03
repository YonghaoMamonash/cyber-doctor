import unittest

from utils.console import to_safe_console_text


class ConsoleSafeTextTests(unittest.TestCase):
    def test_to_safe_console_text_is_gbk_encodable(self):
        raw = "普通文本\u00A0含不间断空格和韩文배틀그라운드"
        safe = to_safe_console_text(raw)
        # Must not raise
        safe.encode("gbk")


if __name__ == "__main__":
    unittest.main()
