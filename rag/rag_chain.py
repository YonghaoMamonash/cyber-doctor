# retrieve类型有很多种,这个文件用于调用不同RAG类型接口
import time
from typing import List
from openai import Stream
from openai.types.chat import ChatCompletionChunk

from rag.query_optimizer import generate_queries, rewrite_question
from rag.retrieval_strategy import choose_multi_query_count, choose_rewrite_enabled
from rag.self_rag import (
    build_refined_queries,
    build_self_rag_eval_prompt,
    parse_self_rag_eval,
    should_retry_retrieval,
)
from rag.retrieve.retrieve_document import retrieve_docs_for_queries
from utils.observability import record_self_rag_eval


def _get_config_value(path: List[str], default):
    try:
        from config.config import Config

        return Config.get_instance().get_with_nested_params(*path)
    except Exception:
        return default


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_rag_prompt(context: str, question: str) -> str:
    if not context:
        return question

    return (
        f"请根据搜索到的文件信息：\n{context}\n"
        f"回答问题：\n{question}\n"
        "若资料不足请明确说明不确定，不要编造。"
    )


def build_self_check_prompt(context: str, question: str, draft_answer: str) -> str:
    return (
        "你是医学问答质检助手。请你检查并修订草稿答案，确保与检索资料一致。\n"
        f"【检索资料】\n{context}\n\n"
        f"【用户问题】\n{question}\n\n"
        f"【草稿答案】\n{draft_answer}\n\n"
        "要求：\n"
        "1) 如果草稿与资料冲突，必须纠正；\n"
        "2) 如果资料不足，明确说明不确定；\n"
        "3) 仅输出最终答案正文，不要输出分析过程。"
    )


def generate_response_with_optional_self_check(
    client,
    prompt: str,
    context: str,
    question: str,
    history: List[List],
    self_check_enabled: bool,
):
    if not self_check_enabled or not context:
        return client.chat_with_ai_stream(prompt, history)

    try:
        draft_answer = client.chat_with_ai(prompt) or ""
        if not draft_answer.strip():
            return client.chat_with_ai_stream(prompt, history)
        review_prompt = build_self_check_prompt(
            context=context, question=question, draft_answer=draft_answer
        )
        return client.chat_with_ai_stream(review_prompt, history)
    except Exception:
        return client.chat_with_ai_stream(prompt, history)


def optimize_context_with_self_rag(
    client,
    question: str,
    rewritten_question: str,
    history: List[List],
    initial_context: str,
    retrieve_context_by_queries,
    self_rag_enabled: bool,
    min_grounded_score: int,
    max_retries: int,
    max_seconds: float,
    max_extra_queries: int,
) -> str:
    if not self_rag_enabled or not initial_context:
        return initial_context

    current_context = initial_context
    start_ts = time.monotonic()
    retries = 0

    while retries < max_retries and (time.monotonic() - start_ts) < max_seconds:
        draft_prompt = build_rag_prompt(context=current_context, question=question)
        draft_answer = client.chat_with_ai(draft_prompt) or ""
        if not draft_answer.strip():
            break

        eval_prompt = build_self_rag_eval_prompt(
            context=current_context,
            question=question,
            draft_answer=draft_answer,
        )
        eval_raw = client.chat_with_ai(eval_prompt) or ""
        eval_result = parse_self_rag_eval(eval_raw)
        retry_triggered = should_retry_retrieval(
            eval_result, min_grounded_score=min_grounded_score
        )
        if not retry_triggered:
            record_self_rag_eval(retry_triggered=False, retry_effective=False)
            break

        refined_queries = build_refined_queries(
            base_question=rewritten_question,
            eval_result=eval_result,
            max_extra_queries=max_extra_queries,
        )
        next_context = retrieve_context_by_queries(refined_queries)
        retry_effective = bool(next_context and next_context != current_context)
        record_self_rag_eval(
            retry_triggered=True,
            retry_effective=retry_effective,
        )
        if not retry_effective:
            break

        current_context = next_context
        retries += 1

    return current_context


def invoke(question: str, history: List[List]) -> Stream[ChatCompletionChunk]:
    history = history or []

    rewrite_enabled = bool(
        _get_config_value(["model", "rag", "query-rewrite", "enabled"], True)
    )
    rewrite_mode = str(
        _get_config_value(["model", "rag", "query-rewrite", "mode"], "always")
    )
    multi_query_enabled = bool(
        _get_config_value(["model", "rag", "multi-query", "enabled"], True)
    )
    multi_query_count = _to_int(
        _get_config_value(["model", "rag", "multi-query", "count"], 3), 3
    )
    top_k_per_query = _to_int(
        _get_config_value(["model", "rag", "retrieval", "top-k-per-query"], 6), 6
    )
    max_context_docs = _to_int(
        _get_config_value(["model", "rag", "retrieval", "max-context-docs"], 8), 8
    )
    query_strategy = str(
        _get_config_value(["model", "rag", "retrieval", "query-strategy"], "auto")
    )
    raptor_lite_enabled = bool(
        _get_config_value(["model", "rag", "raptor-lite", "enabled"], False)
    )
    raptor_summary_top_k = _to_int(
        _get_config_value(["model", "rag", "raptor-lite", "summary-top-k"], 3),
        3,
    )
    self_rag_enabled = bool(
        _get_config_value(["model", "rag", "self-rag", "enabled"], False)
    )
    self_rag_min_grounded_score = _to_int(
        _get_config_value(["model", "rag", "self-rag", "min-grounded-score"], 70),
        70,
    )
    self_rag_max_retries = _to_int(
        _get_config_value(["model", "rag", "self-rag", "max-retries"], 1),
        1,
    )
    self_rag_max_seconds = _to_float(
        _get_config_value(["model", "rag", "self-rag", "max-seconds"], 6),
        6,
    )
    self_rag_max_extra_queries = _to_int(
        _get_config_value(["model", "rag", "self-rag", "max-extra-queries"], 2),
        2,
    )

    rewritten_question = question
    if choose_rewrite_enabled(
        rewrite_enabled=rewrite_enabled,
        rewrite_mode=rewrite_mode,
        question=question,
        history=history,
    ):
        rewritten_question = rewrite_question(question=question, history=history)

    queries = [rewritten_question]
    selected_query_count = choose_multi_query_count(
        question=rewritten_question,
        history=history,
        configured_mode=query_strategy,
        multi_query_enabled=multi_query_enabled,
        multi_query_count=multi_query_count,
    )
    if selected_query_count > 1:
        generated = generate_queries(
            question=rewritten_question,
            history=history,
            count=selected_query_count,
        )
        if rewritten_question not in generated:
            generated.insert(0, rewritten_question)
        queries = generated

    retrieve_fn = None
    if raptor_lite_enabled:
        from model.RAG.retrieve_service import retrieve_with_raptor_lite

        def retrieve_fn(query: str):
            return retrieve_with_raptor_lite(
                query=query,
                summary_top_k=raptor_summary_top_k,
                chunk_top_k=top_k_per_query,
            )

    try:
        _docs, _context = retrieve_docs_for_queries(
            queries=queries,
            top_k_per_query=top_k_per_query,
            max_context_docs=max_context_docs,
            retrieve_fn=retrieve_fn,
        )
    except Exception:
        _context = ""

    answer_self_check_enabled = bool(
        _get_config_value(["model", "rag", "answer-self-check", "enabled"], False)
    )

    from client.clientfactory import Clientfactory

    client = Clientfactory().get_client()

    def _retrieve_context_by_queries(retry_queries: List[str]) -> str:
        try:
            _retry_docs, retry_context = retrieve_docs_for_queries(
                queries=retry_queries,
                top_k_per_query=top_k_per_query,
                max_context_docs=max_context_docs,
                retrieve_fn=retrieve_fn,
            )
            return retry_context
        except Exception:
            return ""

    _context = optimize_context_with_self_rag(
        client=client,
        question=question,
        rewritten_question=rewritten_question,
        history=history,
        initial_context=_context,
        retrieve_context_by_queries=_retrieve_context_by_queries,
        self_rag_enabled=self_rag_enabled,
        min_grounded_score=self_rag_min_grounded_score,
        max_retries=self_rag_max_retries,
        max_seconds=self_rag_max_seconds,
        max_extra_queries=self_rag_max_extra_queries,
    )

    prompt = build_rag_prompt(context=_context, question=question)
    if answer_self_check_enabled and _context:
        prompt += "\n输出前请先检查答案是否与提供资料一致。"

    response = generate_response_with_optional_self_check(
        client=client,
        prompt=prompt,
        context=_context,
        question=question,
        history=history,
        self_check_enabled=answer_self_check_enabled,
    )

    return response
