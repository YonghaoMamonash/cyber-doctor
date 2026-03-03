import unittest

from utils.observability import (
    get_runtime_observability_report,
    record_memory_hit,
    record_planner_action,
    record_self_rag_eval,
    reset_runtime_observability,
)


class ObservabilityTests(unittest.TestCase):
    def test_runtime_report_contains_rates(self):
        reset_runtime_observability()

        record_planner_action("RAG")
        record_planner_action("RAG")
        record_planner_action("InternetSearch")
        record_memory_hit(True)
        record_memory_hit(False)
        record_self_rag_eval(retry_triggered=True, retry_effective=True)
        record_self_rag_eval(retry_triggered=False, retry_effective=False)

        report = get_runtime_observability_report(reset=False)
        self.assertEqual(report["planner"]["total"], 3)
        self.assertEqual(report["planner"]["actions"]["RAG"], 2)
        self.assertAlmostEqual(report["memory"]["hit_rate"], 0.5)
        self.assertAlmostEqual(report["self_rag"]["retry_trigger_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
