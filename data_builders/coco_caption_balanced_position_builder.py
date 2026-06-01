#!/usr/bin/env python3
"""Build balanced 4-position OrderGuard caption matching splits."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


OPTION_LABELS = ["A", "B", "C", "D"]
TASK_TYPE = "caption_matching_semantic_category_hard_negative_balanced_position"
DEFAULT_DEV_SPLIT = "caption_semvis_hard_large_balanced_dev"
DEFAULT_TEST_SPLIT = "caption_semvis_hard_large_balanced_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert 4-position semantic-category hard-negative caption matching "
            "JSONL into balanced 4-position JSONL where every image appears once "
            "at A/B/C/D within each base sample."
        )
    )
    parser.add_argument(
        "--input_dev",
        type=Path,
        default=Path("data/orderguard_caption_semvis_hard_large_dev.jsonl"),
    )
    parser.add_argument(
        "--input_test",
        type=Path,
        default=Path("data/orderguard_caption_semvis_hard_large_test.jsonl"),
    )
    parser.add_argument(
        "--output_dev",
        type=Path,
        default=Path("data/orderguard_caption_semvis_hard_large_balanced_dev.jsonl"),
    )
    parser.add_argument(
        "--output_test",
        type=Path,
        default=Path("data/orderguard_caption_semvis_hard_large_balanced_test.jsonl"),
    )
    parser.add_argument("--project_root", type=Path, default=Path("."))
    return parser.parse_args()


def resolve_project_path(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_by_base_id(rows: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row["base_id"]), []).append(row)
    return grouped


def option_image_set(row: dict[str, Any]) -> set[int]:
    return {int(option["image_id"]) for option in row["options"]}


def find_correct_option(row: dict[str, Any]) -> dict[str, Any] | None:
    correct_options = [option for option in row["options"] if option.get("is_correct")]
    if len(correct_options) != 1:
        return None
    return correct_options[0]


def sorted_options(row: dict[str, Any]) -> list[dict[str, Any]]:
    by_label = {option["label"]: option for option in row["options"]}
    return [by_label[label] for label in OPTION_LABELS]


def relabel_option(option: dict[str, Any], label: str, is_correct: bool) -> dict[str, Any]:
    updated = copy.deepcopy(option)
    updated["label"] = label
    updated["is_correct"] = is_correct
    return updated


def canonical_options_for_base(
    old_base_id: str,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None:
    if len(rows) != 4:
        warn(f"{old_base_id}: expected 4 rows, found {len(rows)}; skipped")
        return None

    positions = [row.get("positive_position") for row in rows]
    if sorted(positions) != OPTION_LABELS:
        warn(f"{old_base_id}: positive_position is not A/B/C/D exactly; skipped")
        return None

    image_sets = [option_image_set(row) for row in rows]
    if any(image_set != image_sets[0] for image_set in image_sets[1:]):
        warn(f"{old_base_id}: option image set is inconsistent across positions; skipped")
        return None

    answer_image_ids = {int(row["answer_image_id"]) for row in rows}
    if len(answer_image_ids) != 1:
        warn(f"{old_base_id}: inconsistent answer_image_id across rows; skipped")
        return None

    for row in rows:
        correct_option = find_correct_option(row)
        if correct_option is None:
            warn(f"{old_base_id}: row has invalid number of correct options; skipped")
            return None
        if int(correct_option["image_id"]) != int(row["answer_image_id"]):
            warn(f"{old_base_id}: correct option does not match answer_image_id; skipped")
            return None

    rows_by_position = {row["positive_position"]: row for row in rows}
    canonical_row = rows_by_position.get("A", rows[0])
    canonical_sorted_options = sorted_options(canonical_row)
    positive_option = find_correct_option(canonical_row)
    if positive_option is None:
        warn(f"{old_base_id}: canonical row has invalid correct option; skipped")
        return None

    negative_options = [
        option for option in canonical_sorted_options if not option.get("is_correct")
    ]
    if len(negative_options) != 3:
        warn(f"{old_base_id}: canonical row does not have 3 negatives; skipped")
        return None

    if len({int(option["image_id"]) for option in [positive_option] + negative_options}) != 4:
        warn(f"{old_base_id}: duplicate images in canonical row; skipped")
        return None

    return positive_option, negative_options, canonical_row


def balanced_layout(
    positive_option: dict[str, Any],
    negative_options: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    n1, n2, n3 = negative_options
    return {
        "A": [positive_option, n1, n2, n3],
        "B": [n3, positive_option, n1, n2],
        "C": [n2, n3, positive_option, n1],
        "D": [n1, n2, n3, positive_option],
    }


def make_row(
    canonical_row: dict[str, Any],
    split: str,
    base_index: int,
    positive_position: str,
    layout_options: list[dict[str, Any]],
) -> dict[str, Any]:
    row = copy.deepcopy(canonical_row)
    base_id = f"{split}_base_{base_index:06d}"
    sample_id = f"{split}_{base_index:06d}_pos_{positive_position}"

    options = []
    for label, option in zip(OPTION_LABELS, layout_options):
        is_correct = int(option["image_id"]) == int(row["answer_image_id"])
        options.append(relabel_option(option, label, is_correct))

    row["sample_id"] = sample_id
    row["base_id"] = base_id
    row["split"] = split
    row["task_type"] = TASK_TYPE
    row["options"] = options
    row["answer"] = positive_position
    row["positive_position"] = positive_position
    return row


def build_balanced_rows(
    input_path: Path,
    split: str,
) -> tuple[list[dict[str, Any]], int, int]:
    source_rows = read_jsonl(input_path)
    grouped = group_by_base_id(source_rows)
    output_rows = []
    skipped = 0
    base_index = 0

    for old_base_id, rows in grouped.items():
        extracted = canonical_options_for_base(old_base_id, rows)
        if extracted is None:
            skipped += 1
            continue

        positive_option, negative_options, canonical_row = extracted
        base_index += 1
        layout = balanced_layout(positive_option, negative_options)
        for positive_position in OPTION_LABELS:
            output_rows.append(
                make_row(
                    canonical_row,
                    split,
                    base_index,
                    positive_position,
                    layout[positive_position],
                )
            )

    return output_rows, base_index, skipped


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def validate_rows(rows: list[dict[str, Any]], project_root: Path) -> None:
    grouped = group_by_base_id(rows)
    for base_id, base_rows in grouped.items():
        assert len(base_rows) == 4, base_id
        assert sorted(row["positive_position"] for row in base_rows) == OPTION_LABELS, base_id

        positions_by_image: dict[int, list[str]] = {}
        positive_positions = []
        positive_image_ids = set()

        for row in base_rows:
            assert len(row["options"]) == 4, row["sample_id"]
            assert row["answer"] == row["positive_position"], row["sample_id"]

            correct_options = [option for option in row["options"] if option["is_correct"]]
            assert len(correct_options) == 1, row["sample_id"]
            assert correct_options[0]["label"] == row["answer"], row["sample_id"]
            assert int(correct_options[0]["image_id"]) == int(row["answer_image_id"]), (
                row["sample_id"]
            )

            positive_positions.append(correct_options[0]["label"])
            positive_image_ids.add(int(correct_options[0]["image_id"]))

            for option in row["options"]:
                path = Path(option["path"])
                assert not path.is_absolute(), option["path"]
                assert (project_root / path).is_file(), option["path"]
                positions_by_image.setdefault(int(option["image_id"]), []).append(
                    option["label"]
                )

        assert len(positions_by_image) == 4, base_id
        for image_id, positions in positions_by_image.items():
            assert sorted(positions) == OPTION_LABELS, f"{base_id}: {image_id}"
        assert len(positive_image_ids) == 1, base_id
        assert sorted(positive_positions) == OPTION_LABELS, base_id


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    input_dev = resolve_project_path(project_root, args.input_dev)
    input_test = resolve_project_path(project_root, args.input_test)
    output_dev = resolve_project_path(project_root, args.output_dev)
    output_test = resolve_project_path(project_root, args.output_test)

    dev_rows, dev_bases, dev_skipped = build_balanced_rows(input_dev, DEFAULT_DEV_SPLIT)
    test_rows, test_bases, test_skipped = build_balanced_rows(input_test, DEFAULT_TEST_SPLIT)

    validate_rows(dev_rows, project_root)
    validate_rows(test_rows, project_root)

    write_jsonl(output_dev, dev_rows)
    write_jsonl(output_test, test_rows)

    print(f"Dev base samples: {dev_bases}")
    print(f"Test base samples: {test_bases}")
    print(f"Dev rows: {len(dev_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Skipped base samples: {dev_skipped + test_skipped}")
    print(f"Skipped dev base samples: {dev_skipped}")
    print(f"Skipped test base samples: {test_skipped}")
    print("All balance checks passed: true")
    print(f"Output dev: {args.output_dev.as_posix()}")
    print(f"Output test: {args.output_test.as_posix()}")


if __name__ == "__main__":
    main()
