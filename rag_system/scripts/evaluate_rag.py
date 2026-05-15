"""
scripts/evaluate_rag.py
────────────────────────
Batch evaluation script for measuring RAG system quality.

Runs a set of question-answer pairs through the pipeline and
measures: critic score, retrieval count, hallucination rate,
latency, and improvement iteration usage.

Usage:
  python scripts/evaluate_rag.py --qa-file eval_qa.json
  python scripts/evaluate_rag.py --qa-file eval_qa.json --output results.json

QA file format (JSON):
  [
    {
      "question": "What is photosynthesis?",
      "expected_keywords": ["sunlight", "chlorophyll", "energy"]
    },
    ...
  ]
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline
from utils.helpers import generate_session_id
from utils.logger import get_logger

logger = get_logger("evaluate_rag")

# Default evaluation set if no QA file is provided
DEFAULT_QA_PAIRS = [
    {
        "question": "What is the main topic of the documents?",
        "expected_keywords": [],
    },
    {
        "question": "Summarize the key points from the knowledge base.",
        "expected_keywords": [],
    },
]


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Calculate fraction of expected keywords present in the answer."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return found / len(keywords)


def run_evaluation(
    qa_pairs: list[dict],
    use_critic: bool = True,
    output_path: str = None,
) -> dict:
    """
    Run all QA pairs through the pipeline and collect metrics.

    Returns:
        Dict with per-question results and aggregate statistics.
    """
    print(f"\n{'═' * 60}")
    print(f"  RAG Evaluation — {len(qa_pairs)} question(s)")
    print(f"{'═' * 60}\n")

    pipeline = RAGPipeline()
    results = []
    session_id = generate_session_id()  # Fresh session for eval

    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        expected_keywords = qa.get("expected_keywords", [])

        print(f"[{i:02d}/{len(qa_pairs):02d}] {question[:70]}")

        try:
            start = time.time()
            result = pipeline.run(
                query=question,
                session_id=session_id,
                use_critic=use_critic,
            )
            elapsed_ms = (time.time() - start) * 1000

            kw_score = keyword_coverage(result.answer, expected_keywords)

            record = {
                "question_id": i,
                "question": question,
                "answer": result.answer,
                "mode": result.mode,
                "critic_score": result.critic_score,
                "critic_passed": result.critic_passed,
                "hallucination_detected": result.hallucination_detected,
                "improvement_iterations": result.improvement_iterations,
                "retrieval_count": result.retrieval_count,
                "latency_ms": elapsed_ms,
                "keyword_coverage": kw_score,
                "sources": [
                    {
                        "source": s["metadata"].get("source"),
                        "score": s["score"],
                    }
                    for s in result.sources[:3]
                ],
                "status": "ok",
            }

            status = "✅" if result.critic_passed else "⚠️"
            hall_flag = " [HALLUCINATION]" if result.hallucination_detected else ""
            print(
                f"       {status} score={result.critic_score:.2f} | "
                f"retrieved={result.retrieval_count} | "
                f"latency={elapsed_ms:.0f}ms{hall_flag}"
            )

        except Exception as e:
            logger.error(f"Evaluation failed for Q{i}: {e}")
            record = {
                "question_id": i,
                "question": question,
                "status": "error",
                "error": str(e),
            }
            print(f"       ❌ ERROR: {e}")

        results.append(record)

    # ── Aggregate statistics ──────────────────────────────────────────────────
    ok_results = [r for r in results if r["status"] == "ok"]
    n = len(ok_results)

    if n > 0:
        avg_critic    = sum(r["critic_score"] for r in ok_results) / n
        avg_latency   = sum(r["latency_ms"] for r in ok_results) / n
        pass_rate     = sum(1 for r in ok_results if r["critic_passed"]) / n
        hall_rate     = sum(1 for r in ok_results if r["hallucination_detected"]) / n
        avg_iters     = sum(r["improvement_iterations"] for r in ok_results) / n
        avg_kw_cov    = sum(r["keyword_coverage"] for r in ok_results) / n
        rag_count     = sum(1 for r in ok_results if r["mode"] == "rag")
    else:
        avg_critic = avg_latency = pass_rate = hall_rate = avg_iters = avg_kw_cov = 0
        rag_count = 0

    summary = {
        "run_at": datetime.utcnow().isoformat(),
        "total_questions": len(qa_pairs),
        "successful": n,
        "failed": len(qa_pairs) - n,
        "rag_mode_count": rag_count,
        "conversational_mode_count": n - rag_count,
        "metrics": {
            "avg_critic_score": round(avg_critic, 3),
            "pass_rate": round(pass_rate, 3),
            "hallucination_rate": round(hall_rate, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_improvement_iterations": round(avg_iters, 2),
            "avg_keyword_coverage": round(avg_kw_cov, 3),
        },
        "results": results,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Evaluation Summary")
    print(f"{'═' * 60}")
    print(f"  Questions   : {len(qa_pairs)}  (ok={n}, failed={len(qa_pairs)-n})")
    print(f"  Pass rate   : {pass_rate*100:.1f}%")
    print(f"  Avg score   : {avg_critic:.3f}")
    print(f"  Hall. rate  : {hall_rate*100:.1f}%")
    print(f"  Avg latency : {avg_latency:.0f}ms")
    print(f"  Avg iters   : {avg_iters:.2f}")
    print(f"  KW coverage : {avg_kw_cov*100:.1f}%")
    print(f"{'═' * 60}\n")

    # Save results
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  📄 Results saved to: {output_path}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system quality")
    parser.add_argument("--qa-file", default=None, help="JSON file with QA pairs")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--no-critic", action="store_true", help="Skip critic evaluation")
    args = parser.parse_args()

    if args.qa_file:
        qa_path = Path(args.qa_file)
        if not qa_path.exists():
            print(f"❌ QA file not found: {args.qa_file}")
            sys.exit(1)
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        print(f"📋 Loaded {len(qa_pairs)} QA pair(s) from: {args.qa_file}")
    else:
        print("📋 No QA file provided — using default evaluation questions.")
        qa_pairs = DEFAULT_QA_PAIRS

    run_evaluation(
        qa_pairs=qa_pairs,
        use_critic=not args.no_critic,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
