ANSWER_FALLBACK_MESSAGE = "抱歉，当前服务暂时不可用，请稍后再试。"


def is_valid_answer_payload(answer) -> bool:
    return isinstance(answer, tuple) and len(answer) >= 2
