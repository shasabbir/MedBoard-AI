"""Command-line entry point for the reproducible evaluation suite."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from medboard.evaluation.runner import run_evaluation, write_evaluation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MedBoard's offline benchmark suite")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results"),
        help="Directory for JSON and Markdown results",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    report = run_evaluation(root)
    write_evaluation_report(report, args.output)
    full = next(
        item for item in report.configurations if item.configuration == "full_medboard"
    )
    print(
        "Evaluation complete: "
        f"{full.metrics.case_count} cases, "
        f"top-3 recall={full.metrics.top_3_recall:.1%}, "
        f"RAG recall@K={report.rag.recall_at_k:.1%}"
    )
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
