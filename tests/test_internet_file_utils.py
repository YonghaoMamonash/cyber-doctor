import unittest

from Internet.file_utils import safe_filename


class InternetFileUtilsTests(unittest.TestCase):
    def test_safe_filename_replaces_invalid_chars(self):
        value = safe_filename('A/B:C*D?"<>|')
        self.assertNotIn("/", value)
        self.assertNotIn(":", value)
        self.assertNotIn("*", value)
        self.assertTrue(value)

    def test_safe_filename_fallback_for_empty(self):
        value = safe_filename("   ")
        self.assertEqual(value, "untitled")

    def test_safe_filename_removes_control_chars(self):
        value = safe_filename("abc\u200e\u200f\u202a")
        self.assertEqual(value, "abc")


if __name__ == "__main__":
    unittest.main()
