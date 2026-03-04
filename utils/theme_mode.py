THEME_MODE_LIGHT = "浅色"
THEME_MODE_DARK = "深色"
THEME_MODE_SYSTEM = "跟随系统"

THEME_MODE_OPTIONS = (
    THEME_MODE_LIGHT,
    THEME_MODE_DARK,
    THEME_MODE_SYSTEM,
)


def normalize_theme_mode(theme_mode: str | None) -> str:
    if theme_mode in THEME_MODE_OPTIONS:
        return theme_mode
    return THEME_MODE_SYSTEM
