import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa.answer import get_answer
from qa.purpose_type import userPurposeType
from qa.question_parser import parse_question
from utils.observability import get_runtime_observability_report


def load_dataset_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_stream_to_text(stream_obj, char_limit: int = 2000) -> str:
    text = ""
    for chunk in stream_obj:
        text += chunk.choices[0].delta.content or ""
        if len(text) >= char_limit:
            break
    return text


def run_rag_inference(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for row in dataset:
        question = row.get("question", "")
        raw_contexts = row.get("contexts", [])
        if not isinstance(raw_contexts, list):
            raw_contexts = [str(raw_contexts)]
        record = {
            "question": question,
            "ground_truth": row.get("ground_truth", ""),
            "contexts": [str(x) for x in raw_contexts if str(x).strip()],
        }
        try:
            qtype = parse_question(question)
            answer = get_answer(question, history=[], question_type=qtype, image_url=None)
            if answer[1] in (
                userPurposeType.text,
                userPurposeType.RAG,
                userPurposeType.KnowledgeGraph,
                userPurposeType.InternetSearch,
            ):
                record["answer"] = _read_stream_to_text(answer[0])
            else:
                record["answer"] = str(answer[0])
            record["error"] = None
            record["question_type"] = str(answer[1])
        except Exception as e:
            record["answer"] = ""
            record["error"] = f"{type(e).__name__}: {e}"
            record["question_type"] = "unknown"

        records.append(record)
    return records


def compute_basic_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    answered = sum(1 for r in records if (r.get("answer") or "").strip())
    errors = sum(1 for r in records if r.get("error"))
    return {
        "total": total,
        "answered": answered,
        "errors": errors,
        "answer_rate": answered / total if total else 0.0,
        "error_rate": errors / total if total else 0.0,
    }


def _normalize_scene(question_type: str) -> str:
    text = (question_type or "").strip()
    if "KnowledgeGraph" in text:
        return "KnowledgeGraph"
    if "InternetSearch" in text:
        return "InternetSearch"
    if "RAG" in text:
        return "RAG"
    return "Other"


def compute_scene_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scenes = defaultdict(
        lambda: {
            "total": 0,
            "answered": 0,
            "errors": 0,
            "answer_rate": 0.0,
            "error_rate": 0.0,
        }
    )

    for name in ("RAG", "KnowledgeGraph", "InternetSearch", "Other"):
        scenes[name]

    for row in records:
        scene = _normalize_scene(str(row.get("question_type", "")))
        scenes[scene]["total"] += 1
        if (row.get("answer") or "").strip():
            scenes[scene]["answered"] += 1
        if row.get("error"):
            scenes[scene]["errors"] += 1

    result = {}
    for scene, agg in scenes.items():
        total = agg["total"]
        answered = agg["answered"]
        errors = agg["errors"]
        result[scene] = {
            **agg,
            "answer_rate": answered / total if total else 0.0,
            "error_rate": errors / total if total else 0.0,
        }
    return result


def infer_failure_reason(record: Dict[str, Any]) -> str | None:
    answer = str(record.get("answer") or "").strip().lower()
    error = str(record.get("error") or "").strip().lower()

    if error:
        retrieval_tokens = (
            "retriev",
            "faiss",
            "向量",
            "knowledge",
            "neo4j",
            "loader",
            "index",
        )
        model_tokens = ("api", "timeout", "network", "connection", "rate limit", "openai")
        if any(token in error for token in retrieval_tokens):
            return "retrieval_error"
        if any(token in error for token in model_tokens):
            return "model_or_network_error"
        return "runtime_error"

    if not answer:
        return "empty_answer"
    if any(token in answer for token in ("资料不足", "信息不足", "不确定", "无法判断")):
        return "insufficient_context"
    if any(token in answer for token in ("冲突", "不一致", "矛盾")):
        return "context_conflict"
    return None


def compute_failure_reason_distribution(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for row in records:
        reason = infer_failure_reason(row)
        if reason is not None:
            counter[reason] += 1
    return dict(counter)


def compute_ragas_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = []
    for row in records:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        ground_truth = str(row.get("ground_truth", "")).strip()
        contexts = row.get("contexts", [])
        if not isinstance(contexts, list):
            contexts = [str(contexts)]
        contexts = [str(x).strip() for x in contexts if str(x).strip()]
        if not (question and answer and ground_truth and contexts):
            continue
        usable.append(
            {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts": contexts,
            }
        )

    if not usable:
        return {"status": "skipped", "note": "缺少 contexts/ground_truth，跳过 RAGAS 计算。"}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": [x["question"] for x in usable],
                "answer": [x["answer"] for x in usable],
                "contexts": [x["contexts"] for x in usable],
                "ground_truth": [x["ground_truth"] for x in usable],
            }
        )
        result = evaluate(
            dataset,
            metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        )
        result_dict = result if isinstance(result, dict) else result.to_dict()
        normalized = {}
        for key, value in result_dict.items():
            try:
                normalized[str(key)] = float(value)
            except Exception:
                normalized[str(key)] = value
        return {"status": "ok", "metrics": normalized, "sample_size": len(usable)}
    except ImportError:
        return {"status": "not_installed", "note": "未安装 ragas/datasets，跳过。"}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def try_ragas_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return compute_ragas_metrics(records)


def write_report(output_path: str, records: List[Dict[str, Any]], metrics: Dict[str, Any]):
    payload = {
        "metrics": metrics,
        "ragas": try_ragas_metrics(records),
        "records": records,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run basic RAG smoke evaluation.")
    parser.add_argument(
        "--dataset",
        default="evaluation/sample_dataset.jsonl",
        help="Path to jsonl dataset.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/latest_report.json",
        help="Path to write evaluation report.",
    )
    args = parser.parse_args()

    dataset = load_dataset_jsonl(args.dataset)
    records = run_rag_inference(dataset)
    metrics = compute_basic_metrics(records)
    metrics["scene"] = compute_scene_metrics(records)
    metrics["failure_reasons"] = compute_failure_reason_distribution(records)
    metrics["runtime_observability"] = get_runtime_observability_report(reset=True)
    write_report(args.output, records, metrics)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
