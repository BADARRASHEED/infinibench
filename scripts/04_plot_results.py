#!/usr/bin/env python
"""Plot task-wise accuracy as a simple bar chart."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover - shown only before installation
    print(f"Missing package: {exc.name}")
    print("Install the project requirements first:")
    print("  pip install -r requirements.txt")
    raise SystemExit(1) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a bar chart from outputs/accuracy_by_task.csv."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/accuracy_by_task.csv"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("outputs/accuracy_by_task.png"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_csv.exists():
        print(f"Accuracy CSV not found: {args.input_csv}")
        print("Run evaluation first:")
        print("  python scripts/03_evaluate_results.py --results-csv data/results_template.csv")
        return 1

    results = pd.read_csv(args.input_csv)
    if results.empty or "accuracy" not in results.columns or "task_type" not in results.columns:
        print("Accuracy CSV does not contain plottable task results.")
        return 1

    results = results.dropna(subset=["accuracy"])
    results = results[results["accuracy"].astype(str).str.strip() != ""]
    if results.empty:
        print("No evaluated accuracy values found. Fill the 'correct' column and rerun evaluation.")
        return 1

    labels = results["task_type"].astype(str).tolist()
    values = results["accuracy"].astype(float) * 100

    fig_width = max(6, len(labels) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    bars = ax.bar(labels, values, color="#4c78a8")

    ax.set_title("Accuracy by Task Type")
    ax.set_xlabel("Task type")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 2, 98),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=150)
    plt.close(fig)

    print(f"Saved chart: {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
