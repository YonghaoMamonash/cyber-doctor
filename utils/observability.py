import threading
from collections import defaultdict
from typing import Dict


class _RuntimeStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(int)

    def inc(self, key: str, value: int = 1):
        if not key:
            return
        with self._lock:
            self._counters[key] += int(value)

    def snapshot(self, reset: bool = False) -> Dict:
        with self._lock:
            counters = dict(self._counters)
            if reset:
                self._counters.clear()

        planner_total = counters.get("planner.total", 0)
        planner_actions = {
            key.split("planner.action.", 1)[1]: value
            for key, value in counters.items()
            if key.startswith("planner.action.")
        }

        memory_total = counters.get("memory.total", 0)
        memory_hit = counters.get("memory.hit", 0)

        self_rag_eval_total = counters.get("self_rag.eval.total", 0)
        self_rag_retry_triggered = counters.get("self_rag.retry.triggered", 0)
        self_rag_retry_effective = counters.get("self_rag.retry.effective", 0)

        return {
            "planner": {
                "total": planner_total,
                "actions": planner_actions,
                "action_distribution": {
                    k: (v / planner_total if planner_total else 0.0)
                    for k, v in planner_actions.items()
                },
            },
            "memory": {
                "total": memory_total,
                "hit": memory_hit,
                "hit_rate": memory_hit / memory_total if memory_total else 0.0,
            },
            "self_rag": {
                "eval_total": self_rag_eval_total,
                "retry_triggered": self_rag_retry_triggered,
                "retry_effective": self_rag_retry_effective,
                "retry_trigger_rate": (
                    self_rag_retry_triggered / self_rag_eval_total
                    if self_rag_eval_total
                    else 0.0
                ),
                "retry_effective_rate": (
                    self_rag_retry_effective / self_rag_retry_triggered
                    if self_rag_retry_triggered
                    else 0.0
                ),
            },
        }


_RUNTIME_STATS = _RuntimeStats()


def record_planner_action(action_name: str):
    action = (action_name or "unknown").strip()
    _RUNTIME_STATS.inc("planner.total", 1)
    _RUNTIME_STATS.inc(f"planner.action.{action}", 1)


def record_memory_hit(hit: bool):
    _RUNTIME_STATS.inc("memory.total", 1)
    if hit:
        _RUNTIME_STATS.inc("memory.hit", 1)


def record_self_rag_eval(retry_triggered: bool, retry_effective: bool):
    _RUNTIME_STATS.inc("self_rag.eval.total", 1)
    if retry_triggered:
        _RUNTIME_STATS.inc("self_rag.retry.triggered", 1)
    if retry_effective:
        _RUNTIME_STATS.inc("self_rag.retry.effective", 1)


def get_runtime_observability_report(reset: bool = False) -> Dict:
    return _RUNTIME_STATS.snapshot(reset=reset)


def reset_runtime_observability():
    _RUNTIME_STATS.snapshot(reset=True)
