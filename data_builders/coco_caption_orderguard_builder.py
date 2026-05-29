#!/usr/bin/env python3
"""Build OrderGuard++ COCO caption matching hard splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


OPTION_LABELS = ["A", "B", "C", "D"]
DEFAULT_SEED = 20260528
TASK_TYPE = "caption_matching"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct 4-image COCO caption matching multiple-choice hard "
            "development and test sets with balanced answer positions."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hard-dev-base-samples", type=int, default=30)
    parser.add_argument("--hard-test-base-samples", type=int, default=100)
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=PROJECT_ROOT / "data/coco/annotations/captions_val2017.json",
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
            "missing COCO caption annotation file: "
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


def build_image_index(coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    if not images_by_id:
        fail("no images found in COCO caption annotations")
    return images_by_id


def build_caption_pool(
    coco: dict[str, Any],
    images_by_id: dict[int, dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    captions_by_image: dict[int, list[dict[str, Any]]] = {}

    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in images_by_id:
            continue

        caption = str(annotation.get("caption", "")).strip()
        if not caption:
            continue

        captions_by_image.setdefault(image_id, []).append(
            {
                "annotation_id": int(annotation["id"]),
                "image_id": image_id,
                "caption": caption,
            }
        )

    if not captions_by_image:
        fail("no usable captions found in COCO caption annotations")

    caption_pool = [
        rng.choice(image_captions)
        for image_captions in captions_by_image.values()
    ]
    rng.shuffle(caption_pool)
    return caption_pool


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
    caption_pool: list[dict[str, Any]],
    caption_offset: int,
    all_image_ids: list[int],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    if total < 0:
        fail("sample counts must be non-negative")

    if caption_offset + total > len(caption_pool):
        fail(
            f"not enough unique caption/image bases for {split}: "
            f"need {caption_offset + total}, found {len(caption_pool)}"
        )

    if len(all_image_ids) < 4:
        fail("at least 4 COCO images are required to build caption matching samples")

    base_samples = []
    selected_captions = caption_pool[caption_offset : caption_offset + total]

    for index, caption_record in enumerate(selected_captions, start=1):
        positive_image_id = int(caption_record["image_id"])
        negative_pool = [image_id for image_id in all_image_ids if image_id != positive_image_id]
        negative_image_ids = rng.sample(negative_pool, 3)
        caption = str(caption_record["caption"])

        base_samples.append(
            {
                "base_id": f"{split}_base_{index:06d}",
                "split": split,
                "task_type": TASK_TYPE,
                "caption": caption,
                "target_category": None,
                "question": f'Which image best matches the caption: "{caption}"?',
                "positive_image_id": positive_image_id,
                "negative_image_ids": negative_image_ids,
            }
        )

    return base_samples, caption_offset + total


def expand_base_samples(
    base_samples: list[dict[str, Any]],
    images_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []

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
                    "task_type": base["task_type"],
                    "target_category": base["target_category"],
                    "caption": base["caption"],
                    "question": base["question"],
                    "options": options,
                    "answer": positive_position,
                    "answer_image_id": int(base["positive_image_id"]),
                    "positive_position": positive_position,
                }
            )

    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    seen_sample_ids: set[str] = set()

    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen_sample_ids:
            fail(f"duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)

        if row["task_type"] != TASK_TYPE:
            fail(f"{sample_id} has invalid task_type: {row['task_type']}")

        if row["caption"] not in row["question"]:
            fail(f"{sample_id} question does not include its caption")

        correct_options = [option for option in row["options"] if option["is_correct"]]
        if len(correct_options) != 1:
            fail(f"{sample_id} has {len(correct_options)} correct options")

        answer_option = correct_options[0]
        if answer_option["label"] != row["answer"]:
            fail(f"{sample_id} answer does not match correct option label")

        if int(answer_option["image_id"]) != int(row["answer_image_id"]):
            fail(f"{sample_id} answer_image_id does not match correct option")

        option_image_ids = [int(option["image_id"]) for option in row["options"]]
        if len(option_image_ids) != len(set(option_image_ids)):
            fail(f"{sample_id} contains duplicate option images")

        for option in row["options"]:
            if not option["is_correct"] and int(option["image_id"]) == int(row["answer_image_id"]):
                fail(f"{sample_id} negative image matches positive image")


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
    images_by_id = build_image_index(coco)
    all_image_ids = sorted(images_by_id)
    caption_pool = build_caption_pool(coco, images_by_id, rng)

    caption_offset = 0
    hard_dev_bases, caption_offset = make_base_samples(
        "hard_dev",
        args.hard_dev_base_samples,
        caption_pool,
        caption_offset,
        all_image_ids,
        rng,
    )
    hard_test_bases, caption_offset = make_base_samples(
        "hard_test",
        args.hard_test_base_samples,
        caption_pool,
        caption_offset,
        all_image_ids,
        rng,
    )

    hard_dev_rows = expand_base_samples(hard_dev_bases, images_by_id)
    hard_test_rows = expand_base_samples(hard_test_bases, images_by_id)

    validate_rows(hard_dev_rows + hard_test_rows)
    verify_image_files(hard_dev_rows + hard_test_rows, image_root)

    hard_dev_path = output_dir / "orderguard_caption_hard_dev.jsonl"
    hard_test_path = output_dir / "orderguard_caption_hard_test.jsonl"
    write_jsonl(hard_dev_path, hard_dev_rows)
    write_jsonl(hard_test_path, hard_test_rows)

    print(f"Wrote {len(hard_dev_rows)} rows to {hard_dev_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {len(hard_test_rows)} rows to {hard_test_path.relative_to(PROJECT_ROOT)}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
