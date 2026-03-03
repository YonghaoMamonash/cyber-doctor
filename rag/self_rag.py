import json
import re
from typing import Dict, List


_SCORE_PATTERN = re.compile(r"(?:score|评分)\s*[:：=]?\s*(\d{1,3})", re.IGNORECASE)


def build_self_rag_eval_prompt(context: str, question: str, draft_answer: str) -> str:
    return (
        "你是检索增强问答的质检器。请评估回答是否被检索资料充分支持。\n"
        f"【检索资料】\n{context}\n\n"
        f"【用户问题】\n{question}\n\n"
        f"【草稿答案】\n{draft_answer}\n\n"
        "请仅输出 JSON："
        '{"score":0-100,"needs_more_retrieval":true/false,"missing_topics":["..."],"reason":"..."}'
    )


def _to_int(value, default: int = 100) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp_score(value: int) -> int:
    return max(0, min(100, value))


def parse_self_rag_eval(raw: str | None) -> Dict:
    result = {
        "score": 100,
        "needs_more_retrieval": False,
        "missing_topics": [],
        "reason": "",
    }
    text = (raw or "").strip()
    if not text:
        return result

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            score = _clamp_score(_to_int(payload.get("score"), 100))
            needs_more = bool(payload.get("needs_more_retrieval", False))
            missing = payload.get("missing_topics", [])
            if not isinstance(missing, list):
                missing = [str(missing)]
            reason = str(payload.get("reason", "")).strip()
            return {
                "score": score,
                "needs_more_retrieval": needs_more,
                "missing_topics": [str(x).strip() for x in missing if str(x).strip()],
                "reason": reason,
            }

    score_match = _SCORE_PATTERN.search(text)
    if score_match:
        result["score"] = _clamp_score(_to_int(score_match.group(1), 100))
    result["reason"] = text[:300]
    return result


def should_retry_retrieval(eval_result: Dict, min_grounded_score: int) -> bool:
    score = _to_int(eval_result.get("score"), 100)
    needs_more = bool(eval_result.get("needs_more_retrieval", False))
    return needs_more or score < int(min_grounded_score)


def build_refined_queries(
    base_question: str,
    eval_result: Dict,
    max_extra_queries: int = 2,
) -> List[str]:
    queries = [base_question]

    missing_topics = eval_result.get("missing_topics", [])
    if not isinstance(missing_topics, list):
        missing_topics = [str(missing_topics)]

    for topic in missing_topics:
        topic = str(topic).strip()
        if not topic:
            continue
        queries.append(f"{base_question} {topic}")
        if len(queries) >= max_extra_queries + 1:
            break

    if len(queries) == 1:
        reason = str(eval_result.get("reason", "")).strip()
        if reason:
            queries.append(f"{base_question} {reason[:40]}")

    deduped = []
    seen = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        deduped.append(q)
    return deduped
