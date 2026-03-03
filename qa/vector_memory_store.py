import json
import os
import threading
import time
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Iterable, List

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


def _flush_ascii(buf: List[str]):
    if not buf:
        return
    token = "".join(buf)
    if token:
        yield token


def _flush_cjk(buf: List[str]):
    if not buf:
        return
    seq = "".join(buf)
    if not seq:
        return
    # Keep phrase-level signal and also yield fine-grained tokens for robust matching.
    yield seq
    for ch in seq:
        yield ch
    if len(seq) >= 2:
        for i in range(len(seq) - 1):
            yield seq[i : i + 2]


def _tokenize(text: str) -> List[str]:
    text = (text or "").strip().lower()
    if not text:
        return []
    ascii_buf: List[str] = []
    cjk_buf: List[str] = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            if ascii_buf:
                yield from _flush_ascii(ascii_buf)
                ascii_buf = []
            cjk_buf.append(ch)
            continue
        if ch.isascii() and ch.isalnum():
            if cjk_buf:
                yield from _flush_cjk(cjk_buf)
                cjk_buf = []
            ascii_buf.append(ch)
            continue
        if ascii_buf:
            yield from _flush_ascii(ascii_buf)
            ascii_buf = []
        if cjk_buf:
            yield from _flush_cjk(cjk_buf)
            cjk_buf = []
    if ascii_buf:
        yield from _flush_ascii(ascii_buf)
    if cjk_buf:
        yield from _flush_cjk(cjk_buf)


def _embed_text(text: str, dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    for token in _tokenize(text):
        h = int(md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec


@dataclass
class _MemoryRecord:
    session_id: str
    text: str
    ts: float
    memory_type: str


class PersistentVectorMemoryStore:
    def __init__(
        self,
        file_path: str,
        max_records: int = 1000,
        vector_dim: int = 256,
    ):
        self._file_path = file_path
        self._max_records = max_records
        self._dim = vector_dim
        self._lock = threading.Lock()
        self._records: List[_MemoryRecord] = []
        self._vectors = np.empty((0, self._dim), dtype=np.float32)
        self._index = None
        self._load()

    def _load(self):
        if not os.path.exists(self._file_path):
            self._ensure_parent_dir()
            return
        rows = []
        with open(self._file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                rows.append(
                    _MemoryRecord(
                        session_id=str(payload.get("session_id", "")),
                        text=str(payload.get("text", "")),
                        ts=float(payload.get("ts", time.time())),
                        memory_type=str(payload.get("memory_type", "explicit")),
                    )
                )
        self._records = rows[-self._max_records :]
        self._rebuild_index()

    def _ensure_parent_dir(self):
        parent = os.path.dirname(self._file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _persist(self):
        self._ensure_parent_dir()
        with open(self._file_path, "w", encoding="utf-8") as f:
            for row in self._records:
                f.write(
                    json.dumps(
                        {
                            "session_id": row.session_id,
                            "text": row.text,
                            "ts": row.ts,
                            "memory_type": row.memory_type,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _rebuild_index(self):
        if not self._records:
            self._vectors = np.empty((0, self._dim), dtype=np.float32)
            self._index = None
            return
        matrix = np.vstack([_embed_text(x.text, self._dim) for x in self._records]).astype(
            np.float32
        )
        self._vectors = matrix
        if faiss is None:
            self._index = None
            return
        index = faiss.IndexFlatIP(self._dim)
        index.add(matrix)
        self._index = index

    def add_facts(self, session_id: str, facts: Iterable[str]):
        if not session_id:
            return
        values = [str(x).strip() for x in facts if str(x).strip()]
        if not values:
            return

        with self._lock:
            existed = {(r.session_id, r.text) for r in self._records}
            for text in values:
                key = (session_id, text)
                if key in existed:
                    continue
                memory_type = "implicit" if text.startswith("偏好:") else "explicit"
                self._records.append(
                    _MemoryRecord(
                        session_id=session_id,
                        text=text,
                        ts=time.time(),
                        memory_type=memory_type,
                    )
                )
                existed.add(key)

            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records :]

            self._rebuild_index()
            self._persist()

    def _fallback_search(self, query: str, top_k: int) -> List[int]:
        query_tokens = set(_tokenize(query))
        scored = []
        for idx, row in enumerate(self._records):
            overlap = len(query_tokens.intersection(set(_tokenize(row.text))))
            scored.append((overlap, row.ts, idx))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [x[2] for x in scored[: max(top_k, 1)]]

    def search_facts(self, session_id: str, query: str, top_k: int = 3) -> List[str]:
        if top_k <= 0 or not self._records:
            return []

        with self._lock:
            if self._index is not None:
                q = _embed_text(query, self._dim).reshape(1, -1).astype(np.float32)
                k = min(max(top_k * 6, top_k), len(self._records))
                _scores, indices = self._index.search(q, k)
                candidate_idx = [int(i) for i in indices[0] if i >= 0]
            else:
                candidate_idx = self._fallback_search(query, top_k * 6)

            session_hits = []
            global_hits = []
            seen = set()
            for idx in candidate_idx:
                if idx >= len(self._records):
                    continue
                row = self._records[idx]
                if row.text in seen:
                    continue
                seen.add(row.text)
                if row.session_id == session_id:
                    session_hits.append(row.text)
                else:
                    global_hits.append(row.text)

            merged = session_hits + global_hits
            return merged[:top_k]


_STORE_CACHE: Dict[str, PersistentVectorMemoryStore] = {}
_STORE_LOCK = threading.Lock()


def get_persistent_memory_store(
    file_path: str,
    max_records: int = 1000,
    vector_dim: int = 256,
) -> PersistentVectorMemoryStore:
    key = os.path.abspath(file_path)
    with _STORE_LOCK:
        if key not in _STORE_CACHE:
            _STORE_CACHE[key] = PersistentVectorMemoryStore(
                file_path=key,
                max_records=max_records,
                vector_dim=vector_dim,
            )
        return _STORE_CACHE[key]
