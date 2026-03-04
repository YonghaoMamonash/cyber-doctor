from utils.chat_ui import format_ai_response_as_cards


def test_format_ai_response_as_cards_assigns_medication_and_risk_icons():
    response = (
        "用药建议：饭后服用阿莫西林，每次500mg。\n\n"
        "风险提醒：若出现皮疹或呼吸困难，请立即停药并就医。"
    )

    html = format_ai_response_as_cards(response)

    assert "💊" in html
    assert "⚠️" in html
    assert html.count('<article class="cyber-result-card">') == 2


def test_format_ai_response_as_cards_handles_empty_text():
    assert format_ai_response_as_cards("") == ""
    assert format_ai_response_as_cards("   ") == ""


def test_format_ai_response_as_cards_splits_single_block_bullets():
    response = "- 饮食建议：控制糖分摄入\n- 复查建议：两周后复诊"

    html = format_ai_response_as_cards(response)

    assert html.count('<article class="cyber-result-card">') == 2
