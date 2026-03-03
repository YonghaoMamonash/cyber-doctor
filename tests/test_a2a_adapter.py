import unittest

from qa.a2a_adapter import A2AHttpAdapter, build_a2a_message_send_payload


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


class A2AAdapterTests(unittest.TestCase):
    def test_build_payload(self):
        payload = build_a2a_message_send_payload(
            text="hello",
            request_id="req-1",
            context_id="ctx-1",
            task_id="task-1",
        )
        self.assertEqual(payload["jsonrpc"], "2.0")
        self.assertEqual(payload["method"], "message/send")
        self.assertEqual(payload["id"], "req-1")
        self.assertEqual(payload["params"]["taskId"], "task-1")

    def test_send_text_success(self):
        def fake_post(_url, json, timeout):
            self.assertEqual(json["method"], "message/send")
            self.assertEqual(timeout, 10)
            return _FakeResponse(
                {
                    "jsonrpc": "2.0",
                    "id": json["id"],
                    "result": {
                        "taskId": "task-1",
                        "state": "completed",
                        "artifacts": [
                            {"parts": [{"type": "text", "text": "delegated answer"}]}
                        ],
                    },
                }
            )

        adapter = A2AHttpAdapter(
            endpoint="http://a2a.local/rpc",
            timeout_seconds=10,
            requester=fake_post,
        )
        result = adapter.send_text("hello", request_id="req-1")
        self.assertTrue(result.success)
        self.assertEqual(result.task_id, "task-1")
        self.assertIn("delegated answer", result.text)

    def test_send_text_jsonrpc_error(self):
        def fake_post(_url, json, timeout):
            return _FakeResponse(
                {
                    "jsonrpc": "2.0",
                    "id": json["id"],
                    "error": {"code": -32000, "message": "downstream unavailable"},
                }
            )

        adapter = A2AHttpAdapter(
            endpoint="http://a2a.local/rpc",
            timeout_seconds=10,
            requester=fake_post,
        )
        result = adapter.send_text("hello", request_id="req-1")
        self.assertFalse(result.success)
        self.assertIn("downstream unavailable", result.error)


if __name__ == "__main__":
    unittest.main()
