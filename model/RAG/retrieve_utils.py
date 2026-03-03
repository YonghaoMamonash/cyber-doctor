import os
import time
from typing import List


def attach_last_modified_metadata(documents: List, getmtime_fn=None) -> List:
    getmtime = getmtime_fn or os.path.getmtime
    for doc in documents:
        source = ""
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            source = doc.metadata.get("source", "")
        if not source:
            continue
        try:
            doc.metadata["last_modified_ts"] = float(getmtime(source))
        except Exception:
            continue
    return documents


def filter_stale_documents(documents: List, stale_days: int, now_ts: float | None = None) -> List:
    if stale_days <= 0:
        return documents

    current_ts = now_ts if now_ts is not None else time.time()
    cutoff = current_ts - stale_days * 24 * 60 * 60

    filtered = []
    for doc in documents:
        modified_ts = None
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            modified_ts = doc.metadata.get("last_modified_ts")
        if modified_ts is None:
            filtered.append(doc)
            continue
        if float(modified_ts) >= cutoff:
            filtered.append(doc)

    return filtered
