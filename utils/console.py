import sys


def to_safe_console_text(value, encoding: str | None = None) -> str:
    text = str(value)
    target_encoding = encoding or sys.stdout.encoding or "utf-8"
    return text.encode(target_encoding, errors="replace").decode(
        target_encoding, errors="replace"
    )


def safe_print(*values):
    print(" ".join(to_safe_console_text(v) for v in values))
