'''根据问答类型选择对应的工具函数进行处理'''
from typing import Tuple, List, Any

from qa.agent_orchestrator import prepare_agent_inputs
from qa.function_tool import map_question_to_function

from qa.purpose_type import userPurposeType


def get_answer(
    question: str, history: List[List | None] = None, question_type=None, image_url=None
) -> Tuple[Any, userPurposeType]:
    """
    根据问题类型调用对应的函数获取结果
    """
    current_history = history or []
    current_type, current_question, current_history = prepare_agent_inputs(
        question=question,
        history=current_history,
        question_type=question_type,
    )

    function = map_question_to_function(current_type)

    args = [current_type, current_question, current_history, image_url]
    result = function(*args)

    return result
