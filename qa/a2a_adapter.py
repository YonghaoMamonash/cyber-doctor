import uuid
from dataclasses import dataclass
from typing import Any, Dict

import requests


def build_a2a_message_send_payload(
    text: str,
    request_id: str | None = None,
    context_id: str | None = None,
    task_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    req_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
    params: Dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": text}],
        }
    }
    if context_id:
        params["contextId"] = context_id
    if task_id:
        params["taskId"] = task_id
    if metadata:
        params["metadata"] = metadata

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "message/send",
        "params": params,
    }


@dataclass
class A2AResult:
    success: bool
    task_id: str
    state: str
    text: str
    raw: Dict[str, Any]
    error: str | None = None


def _extract_text_from_result(result_obj: Dict[str, Any]) -> str:
    if not isinstance(result_obj, dict):
        return ""

    artifacts = result_obj.get("artifacts", [])
    if isinstance(artifacts, list):
        for artifact in artifacts:
            parts = artifact.get("parts", []) if isinstance(artifact, dict) else []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        return text

    history = result_obj.get("history", [])
    if isinstance(history, list):
        for message in reversed(history):
            parts = message.get("parts", []) if isinstance(message, dict) else []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        return text
    return ""


class A2AHttpAdapter:
    def __init__(self, endpoint: str, timeout_seconds: int = 15, requester=None):
        self._endpoint = endpoint
        self._timeout = timeout_seconds
        self._requester = requester or requests.post

    def send_text(
        self,
        text: str,
        request_id: str | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> A2AResult:
        payload = build_a2a_message_send_payload(
            text=text,
            request_id=request_id,
            context_id=context_id,
            task_id=task_id,
            metadata=metadata,
        )
        try:
            resp = self._requester(self._endpoint, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return A2AResult(
                success=False,
                task_id=task_id or "",
                state="failed",
                text="",
                raw={},
                error=f"{type(e).__name__}: {e}",
            )

        if isinstance(data, dict) and data.get("error"):
            err = data.get("error")
            return A2AResult(
                success=False,
                task_id=task_id or "",
                state="failed",
                text="",
                raw=data,
                error=str(err),
            )

        result_obj = data.get("result", {}) if isinstance(data, dict) else {}
        task_id_val = str(result_obj.get("taskId", task_id or ""))
        state = str(result_obj.get("state", "unknown"))
        text_out = _extract_text_from_result(result_obj)
        return A2AResult(
            success=True,
            task_id=task_id_val,
            state=state,
            text=text_out,
            raw=data if isinstance(data, dict) else {"result": data},
            error=None,
        )
