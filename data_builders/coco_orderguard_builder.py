#!/usr/bin/env python3
"""Build OrderGuard++ COCO object VQA order-sensitivity splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


TARGET_CATEGORIES = [
    "dog",
    "cat",
    "car",
    "bicycle",
    "person",
    "bus",
    "train",
    "horse",
    "sheep",
    "bird",
    "boat",
    "chair",
    "couch",
    "dining table",
    "laptop",
    "cell phone",
]

OPTION_LABELS = ["A", "B", "C", "D"]
DEFAULT_SEED = 20260528
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct 4-image COCO object VQA multiple-choice development "
            "and test sets with balanced answer positions."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dev-base-samples", type=int, default=100)
    parser.add_argument("--test-base-samples", type=int, default=300)
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=PROJECT_ROOT / "data/coco/annotations/instances_val2017.json",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=PROJECT_ROOT / "data/coco/val2017",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def require_coco_files(annotation_path: Path, image_root: Path) -> None:
    if not annotation_path.is_file():
        fail(
            "missing COCO annotation file: "
            f"{annotation_path}. Run: bash scripts/download_coco_val2017.sh"
        )

    if not image_root.is_dir():
        fail(
            "missing COCO image directory: "
            f"{image_root}. Run: bash scripts/download_coco_val2017.sh"
        )


def load_coco(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_image_index(coco: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, set[int]], dict[str, int]]:
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    categories_by_name = {
        str(category["name"]): int(category["id"]) for category in coco.get("categories", [])
    }

    categories_by_image: dict[int, set[int]] = {image_id: set() for image_id in images_by_id}
    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id in categories_by_image:
            categories_by_image[image_id].add(category_id)

    missing_categories = [name for name in TARGET_CATEGORIES if name not in categories_by_name]
    if missing_categories:
        fail(f"target categories not found in COCO annotations: {missing_categories}")

    return images_by_id, categories_by_image, categories_by_name


def make_category_pools(
    images_by_id: dict[int, dict[str, Any]],
    categories_by_image: dict[int, set[int]],
    categories_by_name: dict[str, int],
    rng: random.Random,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    image_ids = sorted(images_by_id)
    positives: dict[str, list[int]] = {}
    negatives: dict[str, list[int]] = {}

    for category_name in TARGET_CATEGORIES:
        category_id = categories_by_name[category_name]
        positive_ids = [
            image_id for image_id in image_ids if category_id in categories_by_image[image_id]
        ]
        negative_ids = [
            image_id for image_id in image_ids if category_id not in categories_by_image[image_id]
        ]

        if not positive_ids:
            fail(f"no positive images found for category '{category_name}'")
        if len(negative_ids) < 3:
            fail(f"fewer than 3 negative images found for category '{category_name}'")

        rng.shuffle(positive_ids)
        rng.shuffle(negative_ids)
        positives[category_name] = positive_ids
        negatives[category_name] = negative_ids

    return positives, negatives


def balanced_category_plan(total: int, rng: random.Random) -> list[str]:
    if total < 0:
        fail("sample counts must be non-negative")

    full_repeats, remainder = divmod(total, len(TARGET_CATEGORIES))
    plan = TARGET_CATEGORIES * full_repeats
    plan.extend(rng.sample(TARGET_CATEGORIES, remainder))
    rng.shuffle(plan)
    return plan


def relative_image_path(file_name: str) -> str:
    return f"data/coco/val2017/{file_name}"


def image_record(
    label: str,
    image_id: int,
    images_by_id: dict[int, dict[str, Any]],
    is_correct: bool,
) -> dict[str, Any]:
    image = images_by_id[image_id]
    file_name = str(image["file_name"])
    return {
        "label": label,
        "image_id": image_id,
        "file_name": file_name,
        "path": relative_image_path(file_name),
        "is_correct": is_correct,
    }


def make_base_samples(
    split: str,
    total: int,
    positives: dict[str, list[int]],
    negatives: dict[str, list[int]],
    rng: random.Random,
    positive_offsets: dict[str, int],
) -> list[dict[str, Any]]:
    category_plan = balanced_category_plan(total, rng)
    base_samples: list[dict[str, Any]] = []

    required_by_category: dict[str, int] = defaultdict(int)
    for category_name in category_plan:
        required_by_category[category_name] += 1

    for category_name, required in required_by_category.items():
        current_offset = positive_offsets[category_name]
        available = len(positives[category_name])
        if current_offset + required > available:
            fail(
                f"not enough unique positive images for '{category_name}' in {split}: "
                f"need {current_offset + required}, found {available}"
            )

    for index, category_name in enumerate(category_plan, start=1):
        positive_index = positive_offsets[category_name]
        positive_offsets[category_name] += 1

        positive_image_id = positives[category_name][positive_index]
        negative_image_ids = rng.sample(negatives[category_name], 3)

        base_samples.append(
            {
                "base_id": f"{split}_base_{index:06d}",
                "split": split,
                "target_category": category_name,
                "question": f"Which image contains a {category_name}?",
                "positive_image_id": positive_image_id,
                "negative_image_ids": negative_image_ids,
            }
        )

    return base_samples


def expand_base_samples(
    base_samples: list[dict[str, Any]],
    images_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for base_index, base in enumerate(base_samples, start=1):
        for positive_position in OPTION_LABELS:
            negative_iter = iter(base["negative_image_ids"])
            options = []

            for label in OPTION_LABELS:
                if label == positive_position:
                    image_id = int(base["positive_image_id"])
                    options.append(image_record(label, image_id, images_by_id, True))
                else:
                    image_id = int(next(negative_iter))
                    options.append(image_record(label, image_id, images_by_id, False))

            rows.append(
                {
                    "sample_id": (
                        f"{base['split']}_{base_index:06d}_pos_{positive_position}"
                    ),
                    "base_id": base["base_id"],
                    "split": base["split"],
                    "target_category": base["target_category"],
                    "question": base["question"],
                    "options": options,
                    "answer": positive_position,
                    "answer_image_id": int(base["positive_image_id"]),
                    "positive_position": positive_position,
                }
            )

    return rows


def validate_rows(rows: list[dict[str, Any]], categories_by_image: dict[int, set[int]], categories_by_name: dict[str, int]) -> None:
    seen_sample_ids: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen_sample_ids:
            fail(f"duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)

        category_id = categories_by_name[row["target_category"]]
        correct_options = [option for option in row["options"] if option["is_correct"]]
        if len(correct_options) != 1:
            fail(f"{sample_id} has {len(correct_options)} correct options")

        for option in row["options"]:
            image_categories = categories_by_image[int(option["image_id"])]
            contains_target = category_id in image_categories
            if option["is_correct"] and not contains_target:
                fail(f"{sample_id} positive image does not contain target category")
            if not option["is_correct"] and contains_target:
                fail(f"{sample_id} negative image contains target category")


def verify_image_files(rows: list[dict[str, Any]], image_root: Path) -> None:
    missing = []
    for row in rows:
        for option in row["options"]:
            image_path = image_root / option["file_name"]
            if not image_path.is_file():
                missing.append(str(image_path))
                if len(missing) >= 10:
                    break
        if len(missing) >= 10:
            break

    if missing:
        fail(
            "sampled COCO image files are missing, first missing paths: "
            f"{missing}. Run: bash scripts/download_coco_val2017.sh"
        )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    annotation_path = args.annotation_path.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()

    require_coco_files(annotation_path, image_root)

    rng = random.Random(args.seed)
    coco = load_coco(annotation_path)
    images_by_id, categories_by_image, categories_by_name = build_image_index(coco)
    positives, negatives = make_category_pools(
        images_by_id, categories_by_image, categories_by_name, rng
    )

    positive_offsets: dict[str, int] = defaultdict(int)
    dev_bases = make_base_samples(
        "dev", args.dev_base_samples, positives, negatives, rng, positive_offsets
    )
    test_bases = make_base_samples(
        "test", args.test_base_samples, positives, negatives, rng, positive_offsets
    )

    dev_rows = expand_base_samples(dev_bases, images_by_id)
    test_rows = expand_base_samples(test_bases, images_by_id)

    validate_rows(dev_rows + test_rows, categories_by_image, categories_by_name)
    verify_image_files(dev_rows + test_rows, image_root)

    dev_path = output_dir / "orderguard_dev.jsonl"
    test_path = output_dir / "orderguard_test.jsonl"
    write_jsonl(dev_path, dev_rows)
    write_jsonl(test_path, test_rows)

    print(f"Wrote {len(dev_rows)} rows to {dev_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {len(test_rows)} rows to {test_path.relative_to(PROJECT_ROOT)}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
