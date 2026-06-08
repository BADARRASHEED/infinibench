#!/usr/bin/env python
"""Evaluate manually marked InfiniBench Lite results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - shown only before installation
    print(f"Missing package: {exc.name}")
    print("Install the project requirements first:")
    print("  pip install -r requirements.txt")
    raise SystemExit(1) from exc


TRUE_VALUES = {"1", "1.0", "true", "t", "yes", "y", "correct"}
FALSE_VALUES = {"0", "0.0", "false", "f", "no", "n", "wrong", "incorrect"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate overall and task-wise accuracy from a results CSV."
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("data/results_template.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/accuracy_by_task.csv"),
    )
    return parser.parse_args()


def parse_correct(value: Any) -> float | None:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None
    if text in TRUE_VALUES:
        return 1.0
    if text in FALSE_VALUES:
        return 0.0

    return None


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Results CSV not found: {path}")
        print("Create it first:")
        print("  python scripts/02_create_question_template.py")
        raise SystemExit(1)

    results = pd.read_csv(path, dtype=str).fillna("")
    if results.empty:
        print(f"Results CSV is empty: {path}")
        raise SystemExit(1)

    if "correct" not in results.columns:
        print("The results CSV must contain a 'correct' column.")
        raise SystemExit(1)

    if "task_type" not in results.columns:
        print("No 'task_type' column found. Using 'unknown' for all rows.")
        results["task_type"] = "unknown"

    results["task_type"] = results["task_type"].replace("", "unknown")
    results["_correct_numeric"] = results["correct"].apply(parse_correct)
    return results


def summarize_by_task(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for task_type, group in results.groupby("task_type", dropna=False):
        evaluated = group[group["_correct_numeric"].notna()]
        total_questions = len(group)
        evaluated_questions = len(evaluated)
        correct_answers = int(evaluated["_correct_numeric"].sum()) if evaluated_questions else 0
        accuracy = correct_answers / evaluated_questions if evaluated_questions else None

        rows.append(
            {
                "task_type": str(task_type),
                "total_questions": total_questions,
                "evaluated_questions": evaluated_questions,
                "correct_answers": correct_answers,
                "accuracy": round(accuracy, 4) if accuracy is not None else "",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "task_type",
            "total_questions",
            "evaluated_questions",
            "correct_answers",
            "accuracy",
        ],
    ).sort_values("task_type")


def print_summary(results: pd.DataFrame, by_task: pd.DataFrame) -> None:
    evaluated = results[results["_correct_numeric"].notna()]
    total_questions = len(results)
    evaluated_questions = len(evaluated)
    skipped = total_questions - evaluated_questions

    print("Evaluation summary")
    print("------------------")
    print(f"Total question rows: {total_questions}")
    print(f"Evaluated rows: {evaluated_questions}")
    print(f"Rows without usable correct value: {skipped}")

    if evaluated_questions:
        overall_accuracy = evaluated["_correct_numeric"].mean()
        print(f"Overall accuracy: {overall_accuracy:.2%}")
    else:
        print("Overall accuracy: not available yet")

    print()
    print("Accuracy by task")
    print("----------------")
    for _, row in by_task.iterrows():
        accuracy = row["accuracy"]
        if accuracy == "":
            accuracy_text = "not available"
        else:
            accuracy_text = f"{float(accuracy):.2%}"
        print(
            f"{row['task_type']}: {accuracy_text} "
            f"({row['correct_answers']}/{row['evaluated_questions']} evaluated, "
            f"{row['total_questions']} total)"
        )


def main() -> int:
    args = parse_args()
    results = load_results(args.results_csv)
    by_task = summarize_by_task(results)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    by_task.to_csv(args.output_csv, index=False)

    print_summary(results, by_task)
    print()
    print(f"Saved task-wise accuracy: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
