import json
import re
from typing import Dict, List

_REWRITE_SYSTEM_PROMPT = (
    "你是查询改写助手。请将用户当前问题改写为一个独立、完整、可检索的问题。"
    "只返回改写后的单句问题，不要解释。"
)

_MULTI_QUERY_SYSTEM_PROMPT = (
    "你是检索查询扩展助手。请围绕同一问题给出多个不同视角的检索问题。"
    "只返回 JSON 数组字符串，例如：[\"问题1\",\"问题2\"]。不要输出其他内容。"
)

_LEADING_INDEX_PATTERN = re.compile(r"^\s*\d+\s*[\.\)\-、:：]?\s*")
_VALID_QUERY_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


def _deduplicate_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if not _VALID_QUERY_PATTERN.search(item):
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def parse_queries(raw: str, fallback: str) -> List[str]:
    text = (raw or "").strip()
    if not text:
        return [fallback]

    parsed: List[str] = []

    # Prefer strict JSON array when available.
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            parsed = [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        parsed = []

    if not parsed:
        candidates = re.split(r"[\n;；]+", text)
        cleaned = []
        for candidate in candidates:
            value = _LEADING_INDEX_PATTERN.sub("", candidate).strip().strip("\"'`")
            if value:
                cleaned.append(value)
        parsed = cleaned

    parsed = _deduplicate_keep_order(parsed)
    return parsed or [fallback]


def _get_default_client():
    from client.clientfactory import Clientfactory

    return Clientfactory().get_client()


def _build_rewrite_messages(question: str, history: List[List | None]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": _REWRITE_SYSTEM_PROMPT}]
    for user_input, ai_response in history:
        messages.append({"role": "user", "content": str(user_input)})
        messages.append({"role": "assistant", "content": str(ai_response)})
    messages.append({"role": "user", "content": question})
    return messages


def _build_multi_query_messages(question: str, count: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _MULTI_QUERY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"原始问题：{question}\n"
                f"请给出 {count} 个不同视角的检索问题，保持语义相关且避免重复。"
            ),
        },
    ]


def rewrite_question(
    question: str,
    history: List[List | None] | None = None,
    llm_client=None,
) -> str:
    if not history:
        return question

    client = llm_client or _get_default_client()
    try:
        rewritten = client.chat_using_messages(
            _build_rewrite_messages(question=question, history=history)
        )
        rewritten = (rewritten or "").strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def generate_queries(
    question: str,
    history: List[List | None] | None = None,
    count: int = 3,
    llm_client=None,
) -> List[str]:
    if count <= 1:
        return [question]

    client = llm_client or _get_default_client()
    try:
        response = client.chat_using_messages(
            _build_multi_query_messages(question=question, count=count)
        )
    except Exception:
        return [question]

    queries = parse_queries(response, fallback=question)
    return queries
