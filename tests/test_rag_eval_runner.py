import tempfile
import unittest
import os

from evaluation.rag_eval_runner import (
    compute_basic_metrics,
    compute_ragas_metrics,
    compute_failure_reason_distribution,
    compute_scene_metrics,
    infer_failure_reason,
    load_dataset_jsonl,
)


class RagEvalRunnerTests(unittest.TestCase):
    def test_load_dataset_jsonl(self):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".jsonl", delete=False
        ) as f:
            file_path = f.name
            f.write('{"question":"q1","ground_truth":"a1"}\n')
            f.write('{"question":"q2","ground_truth":"a2"}\n')
        rows = load_dataset_jsonl(file_path)
        os.remove(file_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["question"], "q1")

    def test_compute_basic_metrics(self):
        records = [
            {"answer": "有回答", "error": None},
            {"answer": "", "error": None},
            {"answer": None, "error": "timeout"},
        ]
        metrics = compute_basic_metrics(records)
        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["answered"], 1)
        self.assertEqual(metrics["errors"], 1)

    def test_compute_scene_metrics(self):
        records = [
            {"question_type": "userPurposeType.RAG", "answer": "ok", "error": None},
            {"question_type": "userPurposeType.KnowledgeGraph", "answer": "", "error": None},
            {"question_type": "userPurposeType.InternetSearch", "answer": "", "error": "timeout"},
        ]
        scene = compute_scene_metrics(records)
        self.assertEqual(scene["RAG"]["total"], 1)
        self.assertEqual(scene["KnowledgeGraph"]["total"], 1)
        self.assertEqual(scene["InternetSearch"]["total"], 1)
        self.assertEqual(scene["RAG"]["answered"], 1)
        self.assertEqual(scene["InternetSearch"]["errors"], 1)

    def test_infer_failure_reason(self):
        self.assertEqual(
            infer_failure_reason({"answer": "", "error": "FAISS retrieval failed"}),
            "retrieval_error",
        )
        self.assertEqual(
            infer_failure_reason({"answer": "资料不足，无法确定。", "error": None}),
            "insufficient_context",
        )
        self.assertEqual(
            infer_failure_reason({"answer": "", "error": None}),
            "empty_answer",
        )

    def test_compute_failure_reason_distribution(self):
        records = [
            {"answer": "", "error": "FAISS retrieval failed"},
            {"answer": "资料不足，无法确定。", "error": None},
            {"answer": "", "error": None},
        ]
        dist = compute_failure_reason_distribution(records)
        self.assertEqual(dist["retrieval_error"], 1)
        self.assertEqual(dist["insufficient_context"], 1)
        self.assertEqual(dist["empty_answer"], 1)

    def test_compute_ragas_metrics_without_contexts(self):
        records = [
            {
                "question": "q1",
                "answer": "a1",
                "ground_truth": "g1",
                "contexts": [],
            }
        ]
        result = compute_ragas_metrics(records)
        self.assertIn("status", result)
        self.assertIn(result["status"], ("skipped", "not_installed", "error", "ok"))


if __name__ == "__main__":
    unittest.main()
