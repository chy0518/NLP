#!/usr/bin/env python3
"""Build OrderGuard++ COCO caption matching splits with visual hard negatives."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


OPTION_LABELS = ["A", "B", "C", "D"]
DEFAULT_SEED = 20260528
DEFAULT_TOP_K_HARD_NEGATIVES = 50
TASK_TYPE = "caption_matching_hard_negative"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct 4-image COCO caption matching splits with hard negative "
            "images selected by COCO object-category overlap."
        )
    )
    parser.add_argument("--coco_root", type=Path, default=Path("data/coco"))
    parser.add_argument(
        "--caption_file",
        type=Path,
        default=Path("data/coco/annotations/captions_val2017.json"),
    )
    parser.add_argument(
        "--instance_file",
        type=Path,
        default=Path("data/coco/annotations/instances_val2017.json"),
    )
    parser.add_argument("--out_dir", type=Path, default=Path("data"))
    parser.add_argument("--num_dev", type=int, default=30)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--top_k_hard_negatives",
        type=int,
        default=DEFAULT_TOP_K_HARD_NEGATIVES,
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    if path.is_absolute():
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def require_inputs(coco_root: Path, caption_file: Path, instance_file: Path) -> Path:
    image_root = coco_root / "val2017"

    if not image_root.is_dir():
        fail(
            "missing COCO image directory: "
            f"{image_root}. Run: bash scripts/download_coco_val2017.sh"
        )

    if not caption_file.is_file():
        fail(
            "missing COCO caption annotation file: "
            f"{caption_file}. Run: bash scripts/download_coco_val2017.sh"
        )

    if not instance_file.is_file():
        fail(
            "missing COCO instance annotation file: "
            f"{instance_file}. Run: bash scripts/download_coco_val2017.sh"
        )

    return image_root


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def is_informative_caption(caption: str) -> bool:
    tokens = word_tokens(caption)
    if len(tokens) < 7:
        return False

    content_tokens = [token for token in tokens if token not in STOPWORDS]
    return len(set(content_tokens)) >= 3


def clean_caption(caption: str) -> str:
    return " ".join(str(caption).strip().split())


def build_image_index(caption_coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    images_by_id = {int(image["id"]): image for image in caption_coco.get("images", [])}
    if not images_by_id:
        fail("no images found in COCO caption annotations")
    return images_by_id


def build_category_index(
    instance_coco: dict[str, Any],
    images_by_id: dict[int, dict[str, Any]],
) -> tuple[dict[int, str], dict[int, set[int]], dict[int, set[int]]]:
    category_names = {
        int(category["id"]): str(category["name"])
        for category in instance_coco.get("categories", [])
    }
    if not category_names:
        fail("no categories found in COCO instance annotations")

    image_categories: dict[int, set[int]] = {image_id: set() for image_id in images_by_id}
    images_by_category: dict[int, set[int]] = {category_id: set() for category_id in category_names}

    for annotation in instance_coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in image_categories or category_id not in category_names:
            continue

        image_categories[image_id].add(category_id)
        images_by_category.setdefault(category_id, set()).add(image_id)

    return category_names, image_categories, images_by_category


def category_names_for_ids(category_ids: set[int], category_names: dict[int, str]) -> list[str]:
    return [category_names[category_id] for category_id in sorted(category_ids)]


def relative_image_path(file_name: str, display_coco_root: str) -> str:
    return f"{display_coco_root}/val2017/{file_name}"


def hard_negative_candidates(
    positive_image_id: int,
    image_categories: dict[int, set[int]],
    images_by_category: dict[int, set[int]],
    category_names: dict[int, str],
    top_k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    positive_categories = image_categories[positive_image_id]
    candidate_ids: set[int] = set()

    for category_id in positive_categories:
        candidate_ids.update(images_by_category.get(category_id, set()))

    candidate_ids.discard(positive_image_id)
    candidates = []

    for candidate_image_id in candidate_ids:
        candidate_categories = image_categories.get(candidate_image_id, set())
        shared_category_ids = positive_categories & candidate_categories
        if not shared_category_ids:
            continue

        union_category_ids = positive_categories | candidate_categories
        jaccard = len(shared_category_ids) / len(union_category_ids)
        candidates.append(
            {
                "image_id": candidate_image_id,
                "shared_category_ids": shared_category_ids,
                "shared_count": len(shared_category_ids),
                "jaccard": jaccard,
                "category_count_delta": abs(
                    len(candidate_categories) - len(positive_categories)
                ),
            }
        )

    if len(candidates) < 3:
        return []

    rng.shuffle(candidates)
    candidates.sort(
        key=lambda candidate: (
            candidate["shared_count"],
            candidate["jaccard"],
            -candidate["category_count_delta"],
        ),
        reverse=True,
    )

    top_candidates = candidates[: min(top_k, len(candidates))]
    return [
        {
            "image_id": int(candidate["image_id"]),
            "hard_negative_score": round(float(candidate["jaccard"]), 4),
            "shared_categories": category_names_for_ids(
                candidate["shared_category_ids"], category_names
            ),
        }
        for candidate in top_candidates
    ]


def build_usable_positive_pool(
    caption_coco: dict[str, Any],
    images_by_id: dict[int, dict[str, Any]],
    image_categories: dict[int, set[int]],
    images_by_category: dict[int, set[int]],
    category_names: dict[int, str],
    top_k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    captions_by_image: dict[int, list[dict[str, Any]]] = {}

    for annotation in caption_coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in images_by_id:
            continue

        caption = clean_caption(annotation.get("caption", ""))
        if not caption or not is_informative_caption(caption):
            continue

        captions_by_image.setdefault(image_id, []).append(
            {
                "annotation_id": int(annotation["id"]),
                "caption": caption,
            }
        )

    usable = []
    for image_id in sorted(captions_by_image):
        positive_category_ids = image_categories.get(image_id, set())
        if len(positive_category_ids) < 2:
            continue

        hard_candidates = hard_negative_candidates(
            image_id,
            image_categories,
            images_by_category,
            category_names,
            top_k,
            rng,
        )
        if len(hard_candidates) < 3:
            continue

        caption_record = rng.choice(captions_by_image[image_id])
        usable.append(
            {
                "image_id": image_id,
                "caption": caption_record["caption"],
                "annotation_id": caption_record["annotation_id"],
                "hard_candidates": hard_candidates,
                "positive_categories": category_names_for_ids(
                    positive_category_ids, category_names
                ),
            }
        )

    rng.shuffle(usable)
    return usable


def image_record(
    label: str,
    image_id: int,
    images_by_id: dict[int, dict[str, Any]],
    is_correct: bool,
    display_coco_root: str,
    hard_negative_score: float | None = None,
    shared_categories: list[str] | None = None,
) -> dict[str, Any]:
    image = images_by_id[image_id]
    file_name = str(image["file_name"])
    record: dict[str, Any] = {
        "label": label,
        "image_id": image_id,
        "file_name": file_name,
        "path": relative_image_path(file_name, display_coco_root),
        "is_correct": is_correct,
    }

    if not is_correct:
        record["hard_negative_score"] = hard_negative_score
        record["shared_categories"] = shared_categories or []

    return record


def make_base_samples(
    split: str,
    total: int,
    usable_pool: list[dict[str, Any]],
    pool_offset: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    if total < 0:
        fail("sample counts must be non-negative")

    if pool_offset + total > len(usable_pool):
        fail(
            f"not enough usable positive samples for {split}: "
            f"need {pool_offset + total}, found {len(usable_pool)}"
        )

    base_samples = []
    selected = usable_pool[pool_offset : pool_offset + total]

    for index, positive in enumerate(selected, start=1):
        negative_records = rng.sample(positive["hard_candidates"], 3)
        caption = positive["caption"]

        base_samples.append(
            {
                "base_id": f"{split}_base_{index:06d}",
                "split": split,
                "task_type": TASK_TYPE,
                "caption": caption,
                "target_category": None,
                "question": f'Which image best matches the caption: "{caption}"?',
                "positive_image_id": int(positive["image_id"]),
                "positive_categories": positive["positive_categories"],
                "negative_records": negative_records,
            }
        )

    return base_samples, pool_offset + total


def expand_base_samples(
    base_samples: list[dict[str, Any]],
    images_by_id: dict[int, dict[str, Any]],
    display_coco_root: str,
) -> list[dict[str, Any]]:
    rows = []

    for base_index, base in enumerate(base_samples, start=1):
        for positive_position in OPTION_LABELS:
            negative_iter = iter(base["negative_records"])
            options = []

            for label in OPTION_LABELS:
                if label == positive_position:
                    image_id = int(base["positive_image_id"])
                    options.append(
                        image_record(
                            label,
                            image_id,
                            images_by_id,
                            True,
                            display_coco_root,
                        )
                    )
                else:
                    negative = next(negative_iter)
                    image_id = int(negative["image_id"])
                    options.append(
                        image_record(
                            label,
                            image_id,
                            images_by_id,
                            False,
                            display_coco_root,
                            float(negative["hard_negative_score"]),
                            list(negative["shared_categories"]),
                        )
                    )

            rows.append(
                {
                    "sample_id": (
                        f"{base['split']}_{base_index:06d}_pos_{positive_position}"
                    ),
                    "base_id": base["base_id"],
                    "split": base["split"],
                    "task_type": base["task_type"],
                    "caption": base["caption"],
                    "target_category": base["target_category"],
                    "question": base["question"],
                    "options": options,
                    "answer": positive_position,
                    "answer_image_id": int(base["positive_image_id"]),
                    "positive_position": positive_position,
                    "positive_categories": base["positive_categories"],
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

        if len(row["positive_categories"]) < 2:
            fail(f"{sample_id} positive image has fewer than 2 categories")

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
            if option["is_correct"]:
                continue

            if int(option["image_id"]) == int(row["answer_image_id"]):
                fail(f"{sample_id} negative image matches positive image")

            if not option.get("shared_categories"):
                fail(f"{sample_id} negative image has no shared categories")

            if option.get("hard_negative_score") is None:
                fail(f"{sample_id} negative image is missing hard_negative_score")


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


def average_shared_categories(base_samples: list[dict[str, Any]]) -> float:
    shared_counts = [
        len(negative["shared_categories"])
        for base in base_samples
        for negative in base["negative_records"]
    ]
    if not shared_counts:
        return 0.0
    return sum(shared_counts) / len(shared_counts)


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    if args.top_k_hard_negatives < 3:
        fail("--top_k_hard_negatives must be at least 3")

    coco_root = resolve_project_path(args.coco_root)
    caption_file = resolve_project_path(args.caption_file)
    instance_file = resolve_project_path(args.instance_file)
    out_dir = resolve_project_path(args.out_dir)
    display_coco_root = display_path(args.coco_root)

    image_root = require_inputs(coco_root, caption_file, instance_file)

    rng = random.Random(args.seed)
    caption_coco = load_json(caption_file)
    instance_coco = load_json(instance_file)

    images_by_id = build_image_index(caption_coco)
    category_names, image_categories, images_by_category = build_category_index(
        instance_coco, images_by_id
    )
    usable_pool = build_usable_positive_pool(
        caption_coco,
        images_by_id,
        image_categories,
        images_by_category,
        category_names,
        args.top_k_hard_negatives,
        rng,
    )

    pool_offset = 0
    dev_bases, pool_offset = make_base_samples(
        "caption_hardneg_dev",
        args.num_dev,
        usable_pool,
        pool_offset,
        rng,
    )
    test_bases, pool_offset = make_base_samples(
        "caption_hardneg_test",
        args.num_test,
        usable_pool,
        pool_offset,
        rng,
    )

    dev_rows = expand_base_samples(dev_bases, images_by_id, display_coco_root)
    test_rows = expand_base_samples(test_bases, images_by_id, display_coco_root)

    validate_rows(dev_rows + test_rows)
    verify_image_files(dev_rows + test_rows, image_root)

    dev_path = out_dir / "orderguard_caption_hardneg_dev.jsonl"
    test_path = out_dir / "orderguard_caption_hardneg_test.jsonl"
    write_jsonl(dev_path, dev_rows)
    write_jsonl(test_path, test_rows)

    all_bases = dev_bases + test_bases
    print(f"Usable positive samples: {len(usable_pool)}")
    print(f"Dev base samples: {len(dev_bases)}")
    print(f"Test base samples: {len(test_bases)}")
    print(f"Dev rows: {len(dev_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(
        "Average shared category count per negative: "
        f"{average_shared_categories(all_bases):.2f}"
    )
    print(f"Output dev: {relative_to_project(dev_path)}")
    print(f"Output test: {relative_to_project(test_path)}")
    print(f"Seed: {args.seed}")
    print(f"Top-K hard negatives: {args.top_k_hard_negatives}")


if __name__ == "__main__":
    main()
