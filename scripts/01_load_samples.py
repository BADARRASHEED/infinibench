#!/usr/bin/env python
"""Stream a small set of InfiniBench images from HuggingFace and save them."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    from datasets import load_dataset
    from PIL import Image, UnidentifiedImageError
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - shown only before installation
    print(f"Missing package: {exc.name}")
    print("Install the project requirements first:")
    print("  pip install -r requirements.txt")
    raise SystemExit(1) from exc


DATASET_NAME = "Haoming645/infinibench"
DEFAULT_SPLIT = "train"
IMAGE_KEYS = ("image", "img", "png", "jpg", "jpeg", "image_path")
LABEL_KEYS = ("label", "category", "class", "scene", "prompt", "caption")
ID_KEYS = ("id", "sample_id", "image_id", "file_name", "filename")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream InfiniBench samples from HuggingFace and save PNG images."
    )
    parser.add_argument("--num-samples", type=int, default=50
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/samples"))
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/infinibench_samples.csv"),
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    return parser.parse_args()


def safe_id(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned[:100] or fallback


def csv_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def image_from_value(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value

    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value))

    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            image_path = Path(value["path"])
            if image_path.exists():
                return Image.open(image_path)

    if isinstance(value, str):
        image_path = Path(value)
        if image_path.exists():
            return Image.open(image_path)

    return None


def find_image(sample: dict[str, Any]) -> Image.Image | None:
    for key in IMAGE_KEYS:
        if key in sample:
            image = image_from_value(sample[key])
            if image is not None:
                return image

    for value in sample.values():
        image = image_from_value(value)
        if image is not None:
            return image

    return None


def find_label(sample: dict[str, Any], dataset: Any) -> str:
    label_key = next((key for key in LABEL_KEYS if key in sample), None)
    if label_key is None:
        return ""

    label = sample.get(label_key)
    if label is None:
        return ""

    try:
        features = getattr(dataset, "features", None)
        feature = features.get(label_key) if features is not None else None
        if hasattr(feature, "int2str") and isinstance(label, int):
            return str(feature.int2str(label))
    except Exception:
        pass

    return str(label)


def find_sample_id(sample: dict[str, Any], index: int) -> str:
    fallback = f"sample_{index:05d}"
    for key in ID_KEYS:
        if key in sample and sample[key] not in (None, ""):
            return safe_id(sample[key], fallback)
    return fallback


def save_samples(args: argparse.Namespace) -> int:
    if args.num_samples <= 0:
        print("--num-samples must be greater than 0.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset_name}")
    print(f"Split: {args.split}")
    print("Streaming mode: enabled")

    try:
        dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)
    except Exception as exc:
        print("Could not load the HuggingFace dataset.")
        print("Check your internet connection, dataset name, and split.")
        print(f"Error: {exc}")
        return 1

    rows: list[dict[str, str]] = []
    used_ids: dict[str, int] = {}
    skipped = 0
    iterator = iter(dataset)
    index = 0

    with tqdm(total=args.num_samples, desc="Saving samples", unit="image") as pbar:
        while len(rows) < args.num_samples:
            try:
                sample = next(iterator)
            except StopIteration:
                print("Reached the end of the dataset stream.")
                break
            except Exception as exc:
                skipped += 1
                print(f"Skipping a sample because it could not be read: {exc}")
                continue

            index += 1
            try:
                image = find_image(sample)
                if image is None:
                    raise ValueError("no image field found")

                base_id = find_sample_id(sample, index)
                duplicate_count = used_ids.get(base_id, 0)
                used_ids[base_id] = duplicate_count + 1
                sample_id = base_id if duplicate_count == 0 else f"{base_id}_{duplicate_count + 1}"

                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")

                image_path = args.output_dir / f"{sample_id}.png"
                image.save(image_path, format="PNG")

                rows.append(
                    {
                        "id": sample_id,
                        "image_path": csv_path(image_path),
                        "label": find_label(sample, dataset),
                    }
                )
                pbar.update(1)
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                skipped += 1
                print(f"Skipping sample {index}: {exc}")

    metadata = pd.DataFrame(rows, columns=["id", "image_path", "label"])
    metadata.to_csv(args.metadata_csv, index=False)

    print()
    print(f"Saved images: {len(rows)}")
    print(f"Skipped samples: {skipped}")
    print(f"Metadata CSV: {args.metadata_csv}")

    if not rows:
        print("No images were saved. The dataset fields may have changed.")
        return 1

    return 0


def main() -> int:
    args = parse_args()
    return save_samples(args)


if __name__ == "__main__":
    raise SystemExit(main())
