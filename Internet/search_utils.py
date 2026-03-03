import base64
import re
from urllib.parse import parse_qs, unquote, urlparse

_GENERIC_WORDS = [
    "帮我",
    "请",
    "搜索",
    "搜一下",
    "一下",
    "帮忙",
    "查下",
    "查一下",
    "给我",
]


def extract_real_url(url: str) -> str:
    """Extract the real target URL from common Bing redirect links."""
    if not url:
        return url

    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if "bing.com" not in host:
            return url

        q = parse_qs(parsed.query)
        encoded_u = q.get("u", [None])[0]
        if not encoded_u:
            return url

        # bing commonly wraps target as `u=a1<base64url>`
        if encoded_u.startswith("a1"):
            payload = encoded_u[2:]
            padding = "=" * (-len(payload) % 4)
            try:
                decoded = base64.urlsafe_b64decode(payload + padding).decode("utf-8")
                if decoded.startswith("http"):
                    return decoded
            except Exception:
                pass

            try:
                decoded = unquote(payload)
                if decoded.startswith("http"):
                    return decoded
            except Exception:
                pass

        decoded = unquote(encoded_u)
        if decoded.startswith("http"):
            return decoded
        return url
    except Exception:
        return url


def rank_hits_by_query(hits, query: str, max_items: int = 8):
    if max_items <= 0:
        return []

    token_candidates = [query] + re.split(r"[\s,，。;；:：!?！？]+", query or "")
    tokens = [t for t in token_candidates if t and len(t) >= 2]

    dedup = {}
    for hit in hits or []:
        link = (hit.get("link") or "").strip()
        if not link:
            continue
        if link not in dedup:
            dedup[link] = {
                "title": (hit.get("title") or "").strip(),
                "link": link,
                "snippet": (hit.get("snippet") or "").strip(),
            }

    scored = []
    for hit in dedup.values():
        text = f"{hit['title']} {hit['snippet']}"
        score = 0
        for token in tokens:
            if token in hit["title"]:
                score += 3
            elif token in text:
                score += 1
        scored.append((score, hit))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:max_items]]


def _normalize_query_for_overlap(text: str) -> str:
    value = text or ""
    for word in _GENERIC_WORDS:
        value = value.replace(word, "")
    value = re.sub(r"[\s,，。;；:：!?！？]", "", value)
    return value


def choose_effective_search_question(original_question: str, extracted_question: str) -> str:
    extracted = (extracted_question or "").strip()
    if not extracted:
        return original_question

    origin_norm = _normalize_query_for_overlap(original_question)
    if len(origin_norm) < 2:
        return extracted

    extracted_norm = _normalize_query_for_overlap(extracted)
    if not extracted_norm:
        return original_question

    origin_chars = {c for c in origin_norm if re.match(r"[\u4e00-\u9fffA-Za-z0-9]", c)}
    extracted_chars = {c for c in extracted_norm if re.match(r"[\u4e00-\u9fffA-Za-z0-9]", c)}
    if not origin_chars:
        return extracted

    overlap_ratio = len(origin_chars & extracted_chars) / len(origin_chars)
    if overlap_ratio < 0.2:
        return original_question
    return extracted


def build_snippet_context(hits, max_items: int = 6, max_chars: int = 4000) -> str:
    lines = []
    for idx, hit in enumerate((hits or [])[:max_items], start=1):
        title = hit.get("title", "")
        link = hit.get("link", "")
        snippet = hit.get("snippet", "")
        lines.append(f"[{idx}] 标题: {title}\n链接: {link}\n摘要: {snippet}")
    context = "\n\n".join(lines).strip()
    if len(context) > max_chars:
        return context[:max_chars]
    return context
