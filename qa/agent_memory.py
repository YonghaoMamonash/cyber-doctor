import json
import re
import threading
from typing import Iterable, List, Tuple


_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_SPACE_PATTERN = re.compile(r"\s+")
_NAME_PATTERN = re.compile(r"我叫([^\s，。,.!?！？；;]{1,20})")
_AGE_PATTERN = re.compile(r"我今年(\d{1,3})岁")
_ALLERGY_PATTERN = re.compile(r"我对([^，。,.!?！？；;]{1,30})过敏")
_DISEASE_PATTERNS = [
    re.compile(r"我有([^，。,.!?！？；;]{1,30})"),
    re.compile(r"我患有([^，。,.!?！？；;]{1,30})"),
]
_PREFERENCE_PATTERNS = [
    re.compile(r"我喜欢([^，。,.!?！？；;]{1,30})"),
    re.compile(r"我不喜欢([^，。,.!?！？；;]{1,30})"),
    re.compile(r"我习惯([^，。,.!?！？；;]{1,30})"),
]


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (list, dict, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
    else:
        text = str(value)

    text = _HTML_TAG_PATTERN.sub(" ", text)
    text = _SPACE_PATTERN.sub(" ", text).strip()
    return text


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _normalize_turn(turn) -> Tuple[str, str]:
    if isinstance(turn, (list, tuple)) and len(turn) >= 2:
        return _to_text(turn[0]), _to_text(turn[1])
    return _to_text(turn), ""


def compress_history(
    history: List[List | None] | None,
    keep_recent_turns: int = 8,
    max_message_chars: int = 600,
) -> List[List[str]]:
    if not history:
        return []

    if keep_recent_turns <= 0:
        return []

    selected = history[-keep_recent_turns:]
    compressed: List[List[str]] = []
    for turn in selected:
        user_msg, assistant_msg = _normalize_turn(turn)
        compressed.append(
            [
                _truncate(user_msg, max_message_chars),
                _truncate(assistant_msg, max_message_chars),
            ]
        )
    return compressed


def extract_user_facts(text: str) -> List[str]:
    raw = _to_text(text)
    facts: List[str] = []

    for match in _NAME_PATTERN.findall(raw):
        facts.append(f"姓名: {match}")
    for match in _AGE_PATTERN.findall(raw):
        facts.append(f"年龄: {match}岁")
    for match in _ALLERGY_PATTERN.findall(raw):
        facts.append(f"过敏: {match}")
    for pattern in _DISEASE_PATTERNS:
        for match in pattern.findall(raw):
            facts.append(f"疾病史: {match}")
    for pattern in _PREFERENCE_PATTERNS:
        for match in pattern.findall(raw):
            facts.append(f"偏好: {match}")

    seen = set()
    deduped: List[str] = []
    for item in facts:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _tokenize(text: str) -> set[str]:
    raw = _to_text(text).lower()
    ascii_tokens = re.findall(r"[a-z0-9]+", raw)
    chinese_chars = [ch for ch in raw if "\u4e00" <= ch <= "\u9fff"]
    return set(ascii_tokens + chinese_chars)


class SessionMemoryStore:
    def __init__(self, max_facts_per_session: int = 50):
        self._max_facts = max_facts_per_session
        self._data: dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def add_facts(self, session_id: str, facts: Iterable[str]):
        if not session_id:
            return
        cleaned = [_to_text(x) for x in facts if _to_text(x)]
        if not cleaned:
            return

        with self._lock:
            existing = self._data.get(session_id, [])
            combined = existing + cleaned

            seen = set()
            deduped: List[str] = []
            for item in combined:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)

            if self._max_facts > 0 and len(deduped) > self._max_facts:
                deduped = deduped[-self._max_facts :]

            self._data[session_id] = deduped

    def search_facts(self, session_id: str, query: str, top_k: int = 3) -> List[str]:
        if not session_id or top_k <= 0:
            return []

        with self._lock:
            facts = list(self._data.get(session_id, []))

        if not facts:
            return []

        query_tokens = _tokenize(query)
        scored: List[Tuple[int, int, str]] = []
        for index, fact in enumerate(facts):
            overlap = len(query_tokens.intersection(_tokenize(fact)))
            if overlap > 0:
                scored.append((overlap, -index, fact))

        if not scored:
            return facts[-top_k:]

        scored.sort(reverse=True)
        return [fact for _, _, fact in scored[:top_k]]


def enrich_question_with_memory(
    question: str,
    session_id: str,
    store: SessionMemoryStore,
    top_k: int = 3,
) -> str:
    facts = store.search_facts(session_id=session_id, query=question, top_k=top_k)
    if not facts:
        return question

    lines = "\n".join(f"- {fact}" for fact in facts)
    return (
        "请结合以下历史记忆（仅在相关时使用，不相关则忽略）：\n"
        f"{lines}\n\n"
        f"当前问题：{question}"
    )


DEFAULT_MEMORY_STORE = SessionMemoryStore()


def prepare_memory_context(
    question: str,
    history: List[List | None] | None,
    session_id: str,
    store: SessionMemoryStore | None = None,
    short_term_enabled: bool = True,
    long_term_enabled: bool = True,
    keep_recent_turns: int = 8,
    max_message_chars: int = 600,
    top_k: int = 3,
) -> Tuple[str, List[List[str]]]:
    memory_store = store or DEFAULT_MEMORY_STORE
    compact_history = (
        compress_history(
            history=history,
            keep_recent_turns=keep_recent_turns,
            max_message_chars=max_message_chars,
        )
        if short_term_enabled
        else (history or [])
    )

    if not long_term_enabled:
        return question, compact_history

    memory_store.add_facts(session_id=session_id, facts=extract_user_facts(question))
    enriched_question = enrich_question_with_memory(
        question=question,
        session_id=session_id,
        store=memory_store,
        top_k=top_k,
    )
    return enriched_question, compact_history
