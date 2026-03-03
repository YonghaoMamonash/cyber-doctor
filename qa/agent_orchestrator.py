from typing import List, Tuple

from qa.a2a_adapter import A2AHttpAdapter
from qa.agent_memory import DEFAULT_MEMORY_STORE, prepare_memory_context
from qa.agent_planner import decide_purpose_and_question
from qa.external_ecosystem import (
    DEFAULT_MCP_CATALOG,
    build_external_tool_advice,
    load_mcp_catalog,
)
from qa.purpose_type import userPurposeType
from qa.vector_memory_store import get_persistent_memory_store
from utils.observability import record_memory_hit, record_planner_action


def _get_config_value(path: List[str], default):
    try:
        from config.config import Config

        return Config.get_instance().get_with_nested_params(*path)
    except Exception:
        return default


def _to_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    return default


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _resolve_mcp_catalog(raw_catalog) -> list:
    catalog = load_mcp_catalog(raw_catalog)
    return catalog if catalog else DEFAULT_MCP_CATALOG


def _should_delegate_to_a2a(question: str, min_question_chars: int) -> bool:
    text = (question or "").strip()
    if len(text) < max(min_question_chars, 1):
        return False
    hints = ("综合", "分析", "检索", "文献", "方案", "计划", "比较", "复杂")
    return any(token in text for token in hints) or len(text) >= 20


def delegate_to_a2a(question: str, endpoint: str, timeout_seconds: int = 15) -> str:
    if not endpoint:
        return ""
    adapter = A2AHttpAdapter(endpoint=endpoint, timeout_seconds=timeout_seconds)
    result = adapter.send_text(text=question)
    if not result.success:
        return ""
    return (result.text or "").strip()


def _is_textual_task(purpose: userPurposeType) -> bool:
    return purpose in (
        userPurposeType.text,
        userPurposeType.RAG,
        userPurposeType.KnowledgeGraph,
        userPurposeType.InternetSearch,
    )


def prepare_agent_inputs(
    question: str,
    history: List[List | None] | None,
    question_type: userPurposeType | None,
    llm_client=None,
    memory_store=None,
) -> Tuple[userPurposeType, str, List[List]]:
    current_type = question_type or userPurposeType.text
    current_history = history or []

    if not _is_textual_task(current_type):
        return current_type, question, current_history

    planning_enabled = _to_bool(
        _get_config_value(["model", "agent", "planning", "enabled"], True),
        True,
    )
    llm_decision_enabled = _to_bool(
        _get_config_value(
            ["model", "agent", "planning", "llm-decision-enabled"], True
        ),
        True,
    )
    allow_override_explicit_intent = _to_bool(
        _get_config_value(
            ["model", "agent", "planning", "allow-override-explicit-intent"], False
        ),
        False,
    )

    planned_type, planned_question = decide_purpose_and_question(
        question=question,
        history=current_history,
        original_purpose=current_type,
        llm_client=llm_client,
        planning_enabled=planning_enabled,
        llm_decision_enabled=llm_decision_enabled,
        allow_override_explicit_intent=allow_override_explicit_intent,
    )
    record_planner_action(planned_type.name)

    short_term_enabled = _to_bool(
        _get_config_value(["model", "agent", "memory", "short-term", "enabled"], True),
        True,
    )
    keep_recent_turns = _to_int(
        _get_config_value(["model", "agent", "memory", "short-term", "max-turns"], 8),
        8,
    )
    max_message_chars = _to_int(
        _get_config_value(
            ["model", "agent", "memory", "short-term", "max-message-chars"], 600
        ),
        600,
    )

    long_term_enabled = _to_bool(
        _get_config_value(["model", "agent", "memory", "long-term", "enabled"], True),
        True,
    )
    memory_top_k = _to_int(
        _get_config_value(["model", "agent", "memory", "long-term", "top-k"], 3),
        3,
    )
    long_term_provider = str(
        _get_config_value(["model", "agent", "memory", "long-term", "provider"], "session")
    ).strip().lower()
    if memory_store is not None:
        store = memory_store
    elif long_term_provider == "persistent-vector":
        store = get_persistent_memory_store(
            file_path=str(
                _get_config_value(
                    ["model", "agent", "memory", "long-term", "file-path"],
                    "data/memory/long_term_memory.jsonl",
                )
            ),
            max_records=_to_int(
                _get_config_value(
                    ["model", "agent", "memory", "long-term", "max-records"], 5000
                ),
                5000,
            ),
            vector_dim=_to_int(
                _get_config_value(
                    ["model", "agent", "memory", "long-term", "vector-dim"], 256
                ),
                256,
            ),
        )
    else:
        store = DEFAULT_MEMORY_STORE
    session_id = f"session-{id(current_history)}"
    memory_question, compact_history = prepare_memory_context(
        question=planned_question,
        history=current_history,
        session_id=session_id,
        store=store,
        short_term_enabled=short_term_enabled,
        long_term_enabled=long_term_enabled,
        keep_recent_turns=keep_recent_turns,
        max_message_chars=max_message_chars,
        top_k=memory_top_k,
    )
    if long_term_enabled:
        record_memory_hit(memory_question != planned_question)

    external_enabled = _to_bool(
        _get_config_value(["model", "agent", "external", "tool-advice", "enabled"], False),
        False,
    )
    if external_enabled:
        max_items = _to_int(
            _get_config_value(
                ["model", "agent", "external", "tool-advice", "max-mcp-suggestions"], 3
            ),
            3,
        )
        a2a_enabled = _to_bool(
            _get_config_value(["model", "agent", "external", "a2a", "enabled"], False),
            False,
        )
        catalog = _resolve_mcp_catalog(
            _get_config_value(["model", "agent", "external", "mcp", "catalog"], [])
        )
        advice = build_external_tool_advice(
            question=memory_question,
            purpose=planned_type,
            catalog=catalog,
            max_items=max_items,
            a2a_enabled=a2a_enabled,
        )
        if advice:
            memory_question = f"{memory_question}\n\n{advice}"

    a2a_enabled = _to_bool(
        _get_config_value(["model", "agent", "external", "a2a", "enabled"], False),
        False,
    )
    a2a_mode = str(
        _get_config_value(["model", "agent", "external", "a2a", "mode"], "assist")
    ).strip().lower()
    a2a_min_chars = _to_int(
        _get_config_value(["model", "agent", "external", "a2a", "min-question-chars"], 8),
        8,
    )
    a2a_timeout = _to_int(
        _get_config_value(["model", "agent", "external", "a2a", "timeout-seconds"], 15),
        15,
    )
    a2a_endpoint = str(
        _get_config_value(["model", "agent", "external", "a2a", "endpoint"], "")
    ).strip()
    delegation_probe = question if (question or "").strip() else memory_question
    if a2a_enabled and _should_delegate_to_a2a(delegation_probe, a2a_min_chars):
        delegated_text = delegate_to_a2a(
            question=memory_question,
            endpoint=a2a_endpoint,
            timeout_seconds=a2a_timeout,
        )
        if delegated_text:
            if a2a_mode == "direct":
                memory_question = (
                    f"用户问题：{memory_question}\n"
                    f"外部Agent结果：{delegated_text}\n"
                    "请基于以上结果直接给出最终答复。"
                )
            else:
                memory_question = (
                    f"{memory_question}\n\n外部Agent补充信息（供参考）：\n{delegated_text}"
                )

    return planned_type, memory_question, compact_history
