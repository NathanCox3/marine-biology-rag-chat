import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.ingest import ingest_documents
from app.rag import RAGService


QA_PATH = ROOT / "evals" / "qa_pairs.json"
RESULTS_DIR = ROOT / "evals" / "results"
STORAGE_ROOT = ROOT / "evals" / "storage"


@dataclass(frozen=True)
class EvalRunConfig:
    name: str
    chunk_size: int
    chunk_overlap: int
    similarity_top_k: int
    final_top_k: int


RUNS = [
    EvalRunConfig(
        name="baseline_chunk_900",
        chunk_size=900,
        chunk_overlap=150,
        similarity_top_k=20,
        final_top_k=5,
    ),
    EvalRunConfig(
        name="variant_chunk_450",
        chunk_size=450,
        chunk_overlap=100,
        similarity_top_k=20,
        final_top_k=5,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepEval over the marine biology RAG pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of QA pairs for a smoke run.")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse existing eval vector stores.")
    args = parser.parse_args()

    _load_env()
    _configure_deepeval_runtime()

    qa_pairs = _load_qa_pairs(limit=args.limit)
    judge_model = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o-mini")
    metrics = _build_metrics(judge_model)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    for run_config in RUNS:
        print(f"\n=== Running {run_config.name} ===")
        settings = _settings_for_run(run_config)
        if not args.skip_ingest:
            stats = ingest_documents(settings)
            print(f"Indexed {stats.chunks_indexed} chunks into {stats.storage_path}")

        service = RAGService(settings)
        details = _evaluate_run(service, run_config, qa_pairs, metrics)
        summary = _summarize(details, run_config)
        all_summaries.append(summary)

        detail_path = RESULTS_DIR / f"{run_config.name}.json"
        detail_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
        print(_format_summary(summary))

    comparison = _compare(all_summaries)
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("\n=== Comparison ===")
    print(json.dumps(comparison, indent=2))


def _load_env() -> None:
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT.parent / ".env", override=False)


def _configure_deepeval_runtime() -> None:
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    if not os.getenv("OPENAI_API_KEY") and os.getenv("MARINE_RAG_OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["MARINE_RAG_OPENAI_API_KEY"]
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("DeepEval needs OPENAI_API_KEY or MARINE_RAG_OPENAI_API_KEY.")


def _load_qa_pairs(limit: int | None) -> list[dict[str, Any]]:
    qa_pairs = json.loads(QA_PATH.read_text(encoding="utf-8"))
    if limit is not None:
        return qa_pairs[:limit]
    return qa_pairs


def _settings_for_run(run_config: EvalRunConfig) -> Settings:
    return Settings(
        raw_docs_dir=ROOT / "data" / "raw_docs",
        storage_dir=STORAGE_ROOT / run_config.name,
        chroma_collection=f"marine_biology_docs_{run_config.name}",
        chunk_size=run_config.chunk_size,
        chunk_overlap=run_config.chunk_overlap,
        similarity_top_k=run_config.similarity_top_k,
        final_top_k=run_config.final_top_k,
        enable_reranking=True,
    )


def _build_metrics(judge_model: str) -> dict[str, Any]:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import SingleTurnParams

    return {
        "relevance": AnswerRelevancyMetric(
            threshold=0.7,
            model=judge_model,
            include_reason=True,
            async_mode=False,
        ),
        "faithfulness": FaithfulnessMetric(
            threshold=0.7,
            model=judge_model,
            include_reason=True,
            async_mode=False,
        ),
        "completeness": GEval(
            name="Completeness",
            model=judge_model,
            threshold=0.7,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            criteria=(
                "Score whether the actual answer fully covers the important facts in the expected answer. "
                "Penalize missing key facts, but do not penalize concise wording or extra correct details grounded in context."
            ),
            async_mode=False,
        ),
    }


def _evaluate_run(
    service: RAGService,
    run_config: EvalRunConfig,
    qa_pairs: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    from deepeval.test_case import LLMTestCase

    details: list[dict[str, Any]] = []
    for index, qa in enumerate(qa_pairs, start=1):
        started = time.perf_counter()
        response = service.ask(qa["question"])
        elapsed = time.perf_counter() - started
        retrieval_context = [source.excerpt for source in response.sources]

        test_case = LLMTestCase(
            input=qa["question"],
            actual_output=response.answer,
            expected_output=qa["expected_answer"],
            retrieval_context=retrieval_context,
        )

        metric_results: dict[str, Any] = {}
        for metric_name, metric in metrics.items():
            metric.measure(test_case)
            metric_results[metric_name] = {
                "score": metric.score,
                "passed": bool(metric.is_successful()),
                "reason": metric.reason,
            }

        source_filenames = [source.filename for source in response.sources]
        expected_sources = qa.get("expected_sources", [])
        source_hit = any(source in source_filenames for source in expected_sources)

        item = {
            "run": run_config.name,
            "id": qa["id"],
            "question": qa["question"],
            "expected_answer": qa["expected_answer"],
            "actual_answer": response.answer,
            "expected_sources": expected_sources,
            "retrieved_sources": [source.model_dump() for source in response.sources],
            "source_hit": source_hit,
            "latency_seconds": round(elapsed, 3),
            "metrics": metric_results,
        }
        details.append(item)
        print(
            f"{index:02d}/{len(qa_pairs)} {qa['id']}: "
            f"rel={metric_results['relevance']['score']:.2f} "
            f"faith={metric_results['faithfulness']['score']:.2f} "
            f"comp={metric_results['completeness']['score']:.2f} "
            f"source_hit={source_hit}"
        )

    return details


def _summarize(details: list[dict[str, Any]], run_config: EvalRunConfig) -> dict[str, Any]:
    metric_names = ["relevance", "faithfulness", "completeness"]
    averages = {
        name: round(statistics.mean(item["metrics"][name]["score"] for item in details), 4)
        for name in metric_names
    }
    pass_rates = {
        name: round(
            statistics.mean(1.0 if item["metrics"][name]["passed"] else 0.0 for item in details),
            4,
        )
        for name in metric_names
    }
    return {
        "run": run_config.name,
        "config": {
            "chunk_size": run_config.chunk_size,
            "chunk_overlap": run_config.chunk_overlap,
            "similarity_top_k": run_config.similarity_top_k,
            "final_top_k": run_config.final_top_k,
        },
        "num_questions": len(details),
        "average_scores": averages,
        "pass_rates": pass_rates,
        "source_hit_rate": round(statistics.mean(1.0 if item["source_hit"] else 0.0 for item in details), 4),
        "average_latency_seconds": round(statistics.mean(item["latency_seconds"] for item in details), 4),
        "overall_score": round(statistics.mean(averages.values()), 4),
    }


def _compare(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    baseline, variant = summaries
    deltas = {
        metric: round(variant["average_scores"][metric] - baseline["average_scores"][metric], 4)
        for metric in baseline["average_scores"]
    }
    overall_delta = round(variant["overall_score"] - baseline["overall_score"], 4)
    return {
        "baseline": baseline,
        "variant": variant,
        "change_tested": "Reduced chunk_size from 900 to 450 and chunk_overlap from 150 to 100.",
        "score_deltas": deltas,
        "overall_delta": overall_delta,
        "improved": overall_delta > 0,
    }


def _format_summary(summary: dict[str, Any]) -> str:
    return (
        f"{summary['run']} overall={summary['overall_score']:.4f} "
        f"relevance={summary['average_scores']['relevance']:.4f} "
        f"faithfulness={summary['average_scores']['faithfulness']:.4f} "
        f"completeness={summary['average_scores']['completeness']:.4f} "
        f"source_hit_rate={summary['source_hit_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
