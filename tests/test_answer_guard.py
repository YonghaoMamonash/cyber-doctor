from utils.answer_guard import ANSWER_FALLBACK_MESSAGE, is_valid_answer_payload


def test_is_valid_answer_payload_rejects_none_and_short_tuple():
    assert is_valid_answer_payload(None) is False
    assert is_valid_answer_payload(()) is False
    assert is_valid_answer_payload(("only_one",)) is False


def test_is_valid_answer_payload_accepts_standard_tuple_shape():
    payload = ("stream_or_data", object())
    assert is_valid_answer_payload(payload) is True


def test_answer_fallback_message_non_empty():
    assert isinstance(ANSWER_FALLBACK_MESSAGE, str)
    assert ANSWER_FALLBACK_MESSAGE.strip() != ""
