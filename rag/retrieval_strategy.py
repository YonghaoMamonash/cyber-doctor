from typing import List


_FOLLOWUP_HINTS = (
    "这个",
    "那个",
    "它",
    "他",
    "她",
    "那里",
    "上述",
    "上面",
    "前面",
    "继续",
    "再",
    "然后",
    "该",
    "其",
)

_COMPLEX_HINTS = (
    "并且",
    "同时",
    "分别",
    "以及",
    "和",
    "与",
    "或",
    "还是",
    "比较",
    "对比",
    "区别",
    "优缺点",
    "步骤",
    "方案",
)


def is_followup_question(question: str, history: List[List | None] | None) -> bool:
    if not history:
        return False

    text = (question or "").strip()
    if not text:
        return False

    if len(text) <= 12:
        return True

    return any(token in text for token in _FOLLOWUP_HINTS)


def estimate_query_complexity(question: str) -> int:
    text = (question or "").strip()
    if not text:
        return 0

    separator_score = sum(text.count(ch) for ch in (",", "，", ";", "；", "、"))
    hint_score = sum(1 for token in _COMPLEX_HINTS if token in text)
    return separator_score + hint_score


def choose_rewrite_enabled(
    rewrite_enabled: bool,
    rewrite_mode: str,
    question: str,
    history: List[List | None] | None,
) -> bool:
    if not rewrite_enabled:
        return False

    mode = (rewrite_mode or "always").strip().lower()
    if mode == "followup-only":
        return is_followup_question(question, history)

    return True


def choose_multi_query_count(
    question: str,
    history: List[List | None] | None,
    configured_mode: str,
    multi_query_enabled: bool,
    multi_query_count: int,
) -> int:
    if not multi_query_enabled:
        return 1
    if multi_query_count <= 1:
        return 1

    mode = (configured_mode or "auto").strip().lower()
    if mode == "single":
        return 1
    if mode == "multi":
        return multi_query_count

    if is_followup_question(question, history):
        return multi_query_count
    if estimate_query_complexity(question) >= 2:
        return multi_query_count
    return 1
