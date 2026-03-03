import json
import re
from typing import Dict, List, Tuple

from qa.purpose_type import userPurposeType


_REACT_SYSTEM_PROMPT = (
    "你是医疗问答路由规划器。请使用 ReAct 风格做一步规划，并只输出 JSON。\n"
    "可选 action: text, RAG, KnowledgeGraph, InternetSearch。\n"
    "输出格式: {\"thought\":\"...\",\"action\":\"...\",\"action_input\":\"...\"}\n"
    "要求: action_input 是最终给执行工具的问题文本。"
)

_ACTION_TO_PURPOSE = {
    "TEXT": userPurposeType.text,
    "RAG": userPurposeType.RAG,
    "KNOWLEDGEGRAPH": userPurposeType.KnowledgeGraph,
    "KG": userPurposeType.KnowledgeGraph,
    "INTERNETSEARCH": userPurposeType.InternetSearch,
    "INTERNET": userPurposeType.InternetSearch,
}

_CANONICAL_ACTION = {
    "TEXT": "text",
    "RAG": "RAG",
    "KNOWLEDGEGRAPH": "KnowledgeGraph",
    "KG": "KnowledgeGraph",
    "INTERNETSEARCH": "InternetSearch",
    "INTERNET": "InternetSearch",
}

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _normalize_action(action: str | None) -> str | None:
    if not action:
        return None
    token = re.sub(r"[^A-Za-z]", "", str(action)).upper()
    return _CANONICAL_ACTION.get(token)


def parse_react_plan(raw: str | None) -> Tuple[str | None, str | None]:
    text = (raw or "").strip()
    if not text:
        return None, None

    if not text.startswith("{"):
        match = _JSON_PATTERN.search(text)
        if match:
            text = match.group(0)

    try:
        payload = json.loads(text)
    except Exception:
        return None, None

    if not isinstance(payload, dict):
        return None, None

    action = _normalize_action(payload.get("action"))
    action_input = payload.get("action_input")
    if action_input is None:
        action_input = payload.get("query")

    action_input_text = str(action_input).strip() if action_input else None
    return action, action_input_text


def _build_planner_messages(question: str, history: List[List | None]) -> List[Dict[str, str]]:
    turns = history[-4:] if history else []
    history_text_rows = []
    for idx, turn in enumerate(turns, start=1):
        if isinstance(turn, list) and len(turn) >= 2:
            user_msg = str(turn[0])
            assistant_msg = str(turn[1])
        else:
            user_msg = str(turn)
            assistant_msg = ""
        history_text_rows.append(
            f"轮次{idx} 用户: {user_msg}\n轮次{idx} 助手: {assistant_msg}"
        )
    history_text = "\n".join(history_text_rows) if history_text_rows else "无历史对话"

    user_prompt = (
        f"历史对话：\n{history_text}\n\n"
        f"当前问题：{question}\n"
        "请输出 JSON。"
    )
    return [
        {"role": "system", "content": _REACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _default_llm_client():
    from client.clientfactory import Clientfactory

    return Clientfactory().get_client()


def _is_explicit_intent(purpose: userPurposeType) -> bool:
    return purpose in (
        userPurposeType.RAG,
        userPurposeType.KnowledgeGraph,
        userPurposeType.InternetSearch,
    )


def _rule_based_fallback(
    question: str,
    original_purpose: userPurposeType,
) -> userPurposeType:
    if original_purpose != userPurposeType.text:
        return original_purpose

    text = question.lower()

    if any(keyword in text for keyword in ("知识图谱", "图谱", "关系")):
        return userPurposeType.KnowledgeGraph
    if any(keyword in text for keyword in ("根据知识库", "知识库", "文档", "资料")):
        return userPurposeType.RAG
    if any(keyword in text for keyword in ("搜索", "联网", "实时", "最新", "新闻", "今日", "今天")):
        return userPurposeType.InternetSearch
    return userPurposeType.text


def decide_purpose_and_question(
    question: str,
    history: List[List | None] | None,
    original_purpose: userPurposeType | None,
    llm_client=None,
    planning_enabled: bool = True,
    llm_decision_enabled: bool = True,
    allow_override_explicit_intent: bool = False,
) -> Tuple[userPurposeType, str]:
    purpose = original_purpose or userPurposeType.text
    if not planning_enabled:
        return purpose, question

    if _is_explicit_intent(purpose) and not allow_override_explicit_intent:
        return purpose, question

    fallback_purpose = _rule_based_fallback(question, purpose)
    if not llm_decision_enabled:
        return fallback_purpose, question

    try:
        client = llm_client or _default_llm_client()
        response = client.chat_using_messages(
            _build_planner_messages(question=question, history=history or [])
        )
        action, action_input = parse_react_plan(response)
        if not action:
            return fallback_purpose, question

        token = re.sub(r"[^A-Za-z]", "", action).upper()
        chosen_purpose = _ACTION_TO_PURPOSE.get(token, fallback_purpose)

        if _is_explicit_intent(purpose) and not allow_override_explicit_intent:
            return purpose, question

        return chosen_purpose, (action_input or question)
    except Exception:
        return fallback_purpose, question
