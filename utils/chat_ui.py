import html
import re

_CARD_RULES = [
    (("风险", "禁忌", "副作用", "警惕", "危险", "不良反应", "过敏"), "⚠️", "风险提醒"),
    (("用药", "药物", "剂量", "处方", "服药", "药品"), "💊", "用药建议"),
    (("就医", "急诊", "复诊", "医院", "门诊"), "🏥", "就医建议"),
    (("饮食", "营养", "食物", "膳食"), "🥗", "饮食建议"),
    (("运动", "康复", "锻炼", "训练", "休息"), "🧘", "康复建议"),
]

_DEFAULT_ICON = "🩺"
_DEFAULT_TITLE = "健康建议"


def _normalize_text(text: str) -> str:
    return re.sub(r"\r\n?", "\n", text).strip()


def _split_sections(response_text: str) -> list[str]:
    normalized = _normalize_text(response_text)
    if not normalized:
        return []

    sections = [block.strip() for block in re.split(r"\n{2,}", normalized) if block.strip()]
    if len(sections) != 1:
        return sections

    bullet_sections = []
    for line in normalized.split("\n"):
        raw_line = line.strip()
        if not raw_line:
            continue
        cleaned = re.sub(r"^[-*•\d\.\)\s]+", "", raw_line).strip()
        if cleaned:
            bullet_sections.append(cleaned)

    if len(bullet_sections) >= 2:
        return bullet_sections
    return sections


def _pick_icon_and_title(section: str) -> tuple[str, str]:
    for keywords, icon, title in _CARD_RULES:
        if any(keyword in section for keyword in keywords):
            return icon, title

    heading_match = re.match(r"^([^\n:：]{2,12})[:：]", section)
    if heading_match:
        return _DEFAULT_ICON, heading_match.group(1).strip()

    return _DEFAULT_ICON, _DEFAULT_TITLE


def format_ai_response_as_cards(response_text: str) -> str:
    sections = _split_sections(response_text or "")
    if not sections:
        return ""

    cards = []
    for section in sections:
        icon, title = _pick_icon_and_title(section)
        safe_body = html.escape(section).replace("\n", "<br>")
        safe_title = html.escape(title)
        cards.append(
            f"""
            <article class="cyber-result-card">
                <header class="cyber-result-card-header">
                    <span class="cyber-card-icon">{icon}</span>
                    <span class="cyber-card-title">{safe_title}</span>
                </header>
                <div class="cyber-result-card-body">{safe_body}</div>
            </article>
            """
        )

    return f'<section class="cyber-result-grid">{"".join(cards)}</section>'
