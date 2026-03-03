import re


_INVALID_FILE_CHARS = re.compile(r'[\\/:*?"<>|]')
_CONTROL_CHARS = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\u0000-\u001f]")


def safe_filename(value: str) -> str:
    name = (value or "").strip()
    if not name:
        return "untitled"
    name = _CONTROL_CHARS.sub("", name)
    name = _INVALID_FILE_CHARS.sub("_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:120] or "untitled"
