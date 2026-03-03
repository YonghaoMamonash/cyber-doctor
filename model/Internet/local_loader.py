import os
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from langchain_core.documents import Document


def _parse_html_file(path: str, max_chars: int = 20000) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def load_local_html_documents(base_path: str) -> List[Document]:
    docs: List[Document] = []
    root = Path(base_path)
    if not root.exists():
        return docs

    patterns = ("*.html", "*.htm", "*.mhtml")
    for pattern in patterns:
        for fpath in root.rglob(pattern):
            try:
                content = _parse_html_file(str(fpath))
                if not content.strip():
                    continue
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": str(fpath)},
                    )
                )
            except Exception:
                continue
    return docs
