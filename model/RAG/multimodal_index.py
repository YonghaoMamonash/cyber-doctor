import csv
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
_TABLE_EXTS = {".csv", ".tsv"}


def _source_doc_id(file_path: str, modality: str) -> str:
    payload = f"{modality}:{os.path.abspath(file_path)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _summarize_image_path(file_path: str, max_summary_chars: int) -> str:
    name = Path(file_path).stem.replace("_", " ").replace("-", " ")
    summary = f"图像文件：{name}。建议结合原图进行诊断，不可仅据文件名下结论。"
    return summary[:max_summary_chars]


def _summarize_csv(file_path: str, max_summary_chars: int) -> str:
    rows = []
    headers = []
    try:
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= 2:
                    break
    except Exception:
        return f"表格文件：{Path(file_path).name}"

    header_text = ", ".join(headers) if headers else "无列名"
    sample_lines = []
    for row in rows:
        sample_lines.append(", ".join(row))
    sample_text = " | ".join(sample_lines) if sample_lines else "无样本行"
    summary = (
        f"表格文件：{Path(file_path).name}。"
        f"列：{header_text}。样例：{sample_text}。"
    )
    return summary[:max_summary_chars]


def build_multimodal_documents(
    base_path: str,
    max_summary_chars: int = 260,
) -> Tuple[List[Document], Dict[str, List[Document]]]:
    docs: List[Document] = []
    parent_map: Dict[str, List[Document]] = {}

    for root, _dirs, files in os.walk(base_path):
        for file_name in files:
            ext = Path(file_name).suffix.lower()
            file_path = os.path.join(root, file_name)
            if ext in _IMAGE_EXTS:
                modality = "image"
                summary = _summarize_image_path(file_path, max_summary_chars)
            elif ext in _TABLE_EXTS:
                modality = "table"
                summary = _summarize_csv(file_path, max_summary_chars)
            else:
                continue

            source_id = _source_doc_id(file_path, modality)
            metadata = {
                "source": file_path,
                "source_doc_id": source_id,
                "modality": modality,
                "is_multimodal_summary": True,
                "parent_path": file_path,
            }
            summary_doc = Document(page_content=summary, metadata=metadata)
            docs.append(summary_doc)

            parent_doc = Document(
                page_content=(
                    f"多模态原始文件路径：{file_path}\n"
                    f"模态类型：{modality}\n"
                    f"摘要：{summary}"
                ),
                metadata={
                    "source": file_path,
                    "source_doc_id": source_id,
                    "modality": modality,
                    "is_parent_document": True,
                    "parent_path": file_path,
                },
            )
            parent_map[source_id] = [parent_doc]

    return docs, parent_map
