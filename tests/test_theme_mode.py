from utils.theme_mode import normalize_theme_mode


def test_normalize_theme_mode_accepts_supported_labels():
    assert normalize_theme_mode("浅色") == "浅色"
    assert normalize_theme_mode("深色") == "深色"
    assert normalize_theme_mode("跟随系统") == "跟随系统"


def test_normalize_theme_mode_falls_back_to_system():
    assert normalize_theme_mode("dark") == "跟随系统"
    assert normalize_theme_mode("") == "跟随系统"
    assert normalize_theme_mode(None) == "跟随系统"
