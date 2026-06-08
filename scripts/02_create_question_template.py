#!/usr/bin/env python
"""Create a simple visual spatial reasoning question template CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - shown only before installation
    print(f"Missing package: {exc.name}")
    print("Install the project requirements first:")
    print("  pip install -r requirements.txt")
    raise SystemExit(1) from exc


QUESTION_TEMPLATES = [
    ("counting", "How many chairs are visible in the image?"),
    ("object_presence", "Is there a table visible in the image?"),
    ("clutter_judgment", "Is the scene cluttered?"),
    ("occlusion_judgment", "Are any objects partially occluded?"),
]

OUTPUT_COLUMNS = [
    "id",
    "image_path",
    "label",
    "task_type",
    "question",
    "ground_truth",
    "model_answer",
    "correct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create question and results templates from saved sample metadata."
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=Path("data/infinibench_samples.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/infinibench_questions_template.csv"),
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("data/results_template.csv"),
    )
    return parser.parse_args()


def read_samples(samples_csv: Path) -> pd.DataFrame:
    if not samples_csv.exists():
        print(f"Samples CSV not found: {samples_csv}")
        print("Run this first:")
        print("  python scripts/01_load_samples.py --num-samples 100")
        raise SystemExit(1)

    samples = pd.read_csv(samples_csv, dtype=str).fillna("")
    if samples.empty:
        print(f"Samples CSV is empty: {samples_csv}")
        print("Load samples before creating questions.")
        raise SystemExit(1)

    for column in ("id", "image_path", "label"):
        if column not in samples.columns:
            samples[column] = ""

    return samples


def create_questions(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for index, sample in samples.iterrows():
        sample_id = str(sample.get("id") or f"sample_{index + 1:05d}")
        image_path = str(sample.get("image_path", ""))
        label = str(sample.get("label", ""))

        for task_type, question in QUESTION_TEMPLATES:
            rows.append(
                {
                    "id": f"{sample_id}_{task_type}",
                    "image_path": image_path,
                    "label": label,
                    "task_type": task_type,
                    "question": question,
                    "ground_truth": "",
                    "model_answer": "",
                    "correct": "",
                }
            )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> int:
    args = parse_args()
    samples = read_samples(args.samples_csv)
    questions = create_questions(samples)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    questions.to_csv(args.output_csv, index=False)
    questions.to_csv(args.results_csv, index=False)

    print(f"Loaded samples: {len(samples)}")
    print(f"Created question rows: {len(questions)}")
    print(f"Question template: {args.output_csv}")
    print(f"Editable results template: {args.results_csv}")
    print()
    print("Next step: manually fill ground_truth, model_answer, and correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
