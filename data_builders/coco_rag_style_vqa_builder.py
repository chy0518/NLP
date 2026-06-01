#!/usr/bin/env python3
"""Build RAG-style multi-image VQA splits from COCO val2017."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEXT_LABELS = ["A", "B", "C", "D"]
IMAGE_POSITIONS = [1, 2, 3, 4]
TASK_TYPE = "rag_style_multi_image_vqa"
DEFAULT_SEED = 20260528
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_K_DISTRACTORS = 50

GROUP_CATEGORIES = {
    "animal": [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "vehicle": [
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    ],
    "sports": [
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
    ],
    "food": [
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
    ],
    "furniture": ["chair", "couch", "bed", "dining table", "toilet"],
    "electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "appliance": ["microwave", "oven", "toaster", "sink", "refrigerator"],
}

ACTIVITY_OPTIONS = [
    {"text": "tennis", "categories": ["tennis racket"], "requires_person": True},
    {
        "text": "baseball",
        "categories": ["baseball bat", "baseball glove"],
        "requires_person": True,
    },
    {"text": "skiing", "categories": ["skis"], "requires_person": True},
    {"text": "snowboarding", "categories": ["snowboard"], "requires_person": True},
    {"text": "surfing", "categories": ["surfboard"], "requires_person": True},
    {"text": "kite flying", "categories": ["kite"], "requires_person": True},
    {"text": "skateboarding", "categories": ["skateboard"], "requires_person": True},
    {"text": "cycling", "categories": ["bicycle"], "requires_person": True},
    {
        "text": "riding a motorcycle",
        "categories": ["motorcycle"],
        "requires_person": True,
    },
    {"text": "eating pizza", "categories": ["pizza"], "requires_person": False},
    {"text": "public transportation", "categories": ["bus"], "requires_person": False},
]


@dataclass(frozen=True)
class TextOptionSpec:
    text: str
    category_names: tuple[str, ...]
    category_ids: frozenset[int]


@dataclass(frozen=True)
class TaskSpec:
    vqa_type: str
    question: str
    target_category: str
    answer_text: str
    answer_category_names: tuple[str, ...]
    answer_category_ids: frozenset[int]
    option_pool: tuple[TextOptionSpec, ...]
    group_name: str
    requires_person: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct RAG-style 4-image VQA splits from COCO val2017."
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
    parser.add_argument("--num_dev", type=int, default=50)
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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


def build_image_id_to_file(caption_coco: dict[str, Any]) -> dict[int, str]:
    image_id_to_file = {
        int(image["id"]): str(image["file_name"])
        for image in caption_coco.get("images", [])
    }
    if not image_id_to_file:
        fail("no images found in COCO caption annotations")
    return image_id_to_file


def build_image_id_to_captions(caption_coco: dict[str, Any]) -> dict[int, list[str]]:
    image_id_to_captions: dict[int, list[str]] = {}
    for annotation in caption_coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        caption = " ".join(str(annotation.get("caption", "")).strip().split())
        if caption:
            image_id_to_captions.setdefault(image_id, []).append(caption)
    return image_id_to_captions


def build_instance_indexes(
    instance_coco: dict[str, Any],
    image_id_to_file: dict[int, str],
) -> tuple[
    dict[int, str],
    dict[str, int],
    dict[int, str],
    dict[int, set[int]],
    dict[int, set[int]],
]:
    category_id_to_name = {
        int(category["id"]): str(category["name"])
        for category in instance_coco.get("categories", [])
    }
    name_to_category_id = {name: category_id for category_id, name in category_id_to_name.items()}
    category_id_to_supergroup = {
        int(category["id"]): str(category.get("supercategory", ""))
        for category in instance_coco.get("categories", [])
    }
    if not category_id_to_name:
        fail("no categories found in COCO instance annotations")

    image_id_to_categories: dict[int, set[int]] = {
        image_id: set() for image_id in image_id_to_file
    }
    images_by_category: dict[int, set[int]] = {
        category_id: set() for category_id in category_id_to_name
    }

    for annotation in instance_coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in image_id_to_categories or category_id not in category_id_to_name:
            continue
        image_id_to_categories[image_id].add(category_id)
        images_by_category.setdefault(category_id, set()).add(image_id)

    return (
        category_id_to_name,
        name_to_category_id,
        category_id_to_supergroup,
        image_id_to_categories,
        images_by_category,
    )


def category_names_for_ids(
    category_ids: set[int] | frozenset[int],
    category_id_to_name: dict[int, str],
) -> list[str]:
    return [category_id_to_name[category_id] for category_id in sorted(category_ids)]


def available_category_names(
    names: list[str],
    name_to_category_id: dict[str, int],
) -> list[str]:
    return [name for name in names if name in name_to_category_id]


def make_option_specs(
    category_names: list[str],
    name_to_category_id: dict[str, int],
) -> tuple[TextOptionSpec, ...]:
    specs = []
    for name in category_names:
        category_id = name_to_category_id[name]
        specs.append(
            TextOptionSpec(
                text=name,
                category_names=(name,),
                category_ids=frozenset({category_id}),
            )
        )
    return tuple(specs)


def build_activity_option_specs(
    name_to_category_id: dict[str, int],
) -> tuple[TextOptionSpec, ...]:
    options = []
    for option in ACTIVITY_OPTIONS:
        category_names = available_category_names(option["categories"], name_to_category_id)
        if not category_names:
            continue
        options.append(
            TextOptionSpec(
                text=option["text"],
                category_names=tuple(category_names),
                category_ids=frozenset(name_to_category_id[name] for name in category_names),
            )
        )
    return tuple(options)


def build_task_specs(name_to_category_id: dict[str, int]) -> list[TaskSpec]:
    specs: list[TaskSpec] = []

    for group_name, group_categories in GROUP_CATEGORIES.items():
        available_names = available_category_names(group_categories, name_to_category_id)
        if len(available_names) < 4:
            continue

        option_pool = make_option_specs(available_names, name_to_category_id)
        for category_name in available_names:
            category_id = name_to_category_id[category_name]
            specs.append(
                TaskSpec(
                    vqa_type="object_existence",
                    question="Which object is present in the relevant image?",
                    target_category=category_name,
                    answer_text=category_name,
                    answer_category_names=(category_name,),
                    answer_category_ids=frozenset({category_id}),
                    option_pool=option_pool,
                    group_name=group_name,
                )
            )

    animal_names = available_category_names(GROUP_CATEGORIES["animal"], name_to_category_id)
    if len(animal_names) >= 4:
        animal_options = make_option_specs(animal_names, name_to_category_id)
        for category_name in animal_names:
            category_id = name_to_category_id[category_name]
            specs.append(
                TaskSpec(
                    vqa_type="animal_type",
                    question="Which type of animal appears in the relevant image?",
                    target_category=category_name,
                    answer_text=category_name,
                    answer_category_names=(category_name,),
                    answer_category_ids=frozenset({category_id}),
                    option_pool=animal_options,
                    group_name="animal",
                )
            )

    vehicle_names = available_category_names(GROUP_CATEGORIES["vehicle"], name_to_category_id)
    if len(vehicle_names) >= 4:
        vehicle_options = make_option_specs(vehicle_names, name_to_category_id)
        for category_name in vehicle_names:
            category_id = name_to_category_id[category_name]
            specs.append(
                TaskSpec(
                    vqa_type="vehicle_type",
                    question="Which type of vehicle appears in the relevant image?",
                    target_category=category_name,
                    answer_text=category_name,
                    answer_category_names=(category_name,),
                    answer_category_ids=frozenset({category_id}),
                    option_pool=vehicle_options,
                    group_name="vehicle",
                )
            )

    activity_options = build_activity_option_specs(name_to_category_id)
    activity_by_text = {option.text: option for option in activity_options}
    for activity in ACTIVITY_OPTIONS:
        if activity["text"] not in activity_by_text:
            continue
        answer_option = activity_by_text[activity["text"]]
        for category_name in answer_option.category_names:
            specs.append(
                TaskSpec(
                    vqa_type="activity",
                    question="What activity is most likely shown in the relevant image?",
                    target_category=category_name,
                    answer_text=answer_option.text,
                    answer_category_names=answer_option.category_names,
                    answer_category_ids=answer_option.category_ids,
                    option_pool=activity_options,
                    group_name="activity",
                    requires_person=bool(activity.get("requires_person", False)),
                )
            )

    return specs


def text_option_candidates(
    spec: TaskSpec,
    positive_category_ids: set[int],
    rng: random.Random,
) -> list[TextOptionSpec]:
    candidates = [
        option
        for option in spec.option_pool
        if option.text != spec.answer_text and option.category_ids.isdisjoint(positive_category_ids)
    ]
    if len(candidates) < 3:
        return []
    rng.shuffle(candidates)
    return candidates[:3]


def category_overlap_score(positive_categories: set[int], negative_categories: set[int]) -> float:
    union_categories = positive_categories | negative_categories
    if not union_categories:
        return 0.0
    return len(positive_categories & negative_categories) / len(union_categories)


def same_supergroup_bonus(
    target_category_id: int,
    negative_categories: set[int],
    preferred_distractor_category_ids: set[int],
    category_id_to_supergroup: dict[int, str],
) -> float:
    target_supergroup = category_id_to_supergroup.get(target_category_id)
    if not target_supergroup:
        return 0.0

    preferred_same_group = [
        category_id
        for category_id in preferred_distractor_category_ids
        if category_id_to_supergroup.get(category_id) == target_supergroup
    ]
    if negative_categories.intersection(preferred_same_group):
        return 0.3

    if any(
        category_id_to_supergroup.get(category_id) == target_supergroup
        for category_id in negative_categories
    ):
        return 0.3
    return 0.0


def person_cooccurrence_bonus(
    positive_categories: set[int],
    negative_categories: set[int],
    person_category_id: int | None,
) -> float:
    if person_category_id is None:
        return 0.0
    if person_category_id in positive_categories and person_category_id in negative_categories:
        return 0.2
    return 0.0


def initial_candidate_ids(
    positive_categories: set[int],
    forbidden_correct_category_ids: set[int],
    preferred_distractor_category_ids: set[int],
    images_by_category: dict[int, set[int]],
) -> set[int]:
    candidate_ids: set[int] = set()
    for category_id in positive_categories - forbidden_correct_category_ids:
        candidate_ids.update(images_by_category.get(category_id, set()))
    for category_id in preferred_distractor_category_ids:
        candidate_ids.update(images_by_category.get(category_id, set()))
    return candidate_ids


def hard_distractor_candidates(
    relevant_image_id: int,
    target_category_id: int,
    positive_categories: set[int],
    forbidden_correct_category_ids: set[int],
    wrong_options: list[TextOptionSpec],
    image_id_to_categories: dict[int, set[int]],
    images_by_category: dict[int, set[int]],
    category_id_to_name: dict[int, str],
    category_id_to_supergroup: dict[int, str],
    person_category_id: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    preferred_distractor_category_ids = set().union(
        *(set(option.category_ids) for option in wrong_options)
    )
    candidate_ids = initial_candidate_ids(
        positive_categories,
        forbidden_correct_category_ids,
        preferred_distractor_category_ids,
        images_by_category,
    )
    candidate_ids.discard(relevant_image_id)

    candidates = []
    for candidate_image_id in candidate_ids:
        negative_categories = image_id_to_categories.get(candidate_image_id, set())
        if not negative_categories:
            continue
        if negative_categories.intersection(forbidden_correct_category_ids):
            continue

        shared_category_ids = positive_categories & negative_categories
        overlap = category_overlap_score(positive_categories, negative_categories)
        supergroup_bonus = same_supergroup_bonus(
            target_category_id,
            negative_categories,
            preferred_distractor_category_ids,
            category_id_to_supergroup,
        )
        person_bonus = person_cooccurrence_bonus(
            positive_categories, negative_categories, person_category_id
        )
        hard_score = overlap + supergroup_bonus + person_bonus

        has_preferred_distractor = bool(
            negative_categories.intersection(preferred_distractor_category_ids)
        )
        if not shared_category_ids and not has_preferred_distractor and supergroup_bonus == 0:
            continue

        candidates.append(
            {
                "image_id": candidate_image_id,
                "hard_negative_score": round(hard_score, 4),
                "category_overlap_score": round(overlap, 4),
                "same_supergroup_bonus": round(supergroup_bonus, 4),
                "person_cooccurrence_bonus": round(person_bonus, 4),
                "shared_categories": category_names_for_ids(
                    shared_category_ids, category_id_to_name
                ),
                "has_preferred_distractor": has_preferred_distractor,
            }
        )

    rng.shuffle(candidates)
    candidates.sort(
        key=lambda candidate: (
            candidate["hard_negative_score"],
            candidate["has_preferred_distractor"],
            len(candidate["shared_categories"]),
            candidate["category_overlap_score"],
        ),
        reverse=True,
    )
    return candidates


def build_candidate_pool(
    task_specs: list[TaskSpec],
    image_id_to_file: dict[int, str],
    image_id_to_categories: dict[int, set[int]],
    images_by_category: dict[int, set[int]],
    category_id_to_name: dict[int, str],
    category_id_to_supergroup: dict[int, str],
    name_to_category_id: dict[str, int],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    pool = []
    skipped = Counter()
    person_category_id = name_to_category_id.get("person")

    shuffled_specs = list(task_specs)
    rng.shuffle(shuffled_specs)

    for spec in shuffled_specs:
        target_category_id = name_to_category_id[spec.target_category]
        target_image_ids = list(images_by_category.get(name_to_category_id[spec.target_category], set()))
        rng.shuffle(target_image_ids)

        for image_id in target_image_ids:
            if image_id not in image_id_to_file:
                skipped["missing_image_record"] += 1
                continue

            positive_categories = image_id_to_categories.get(image_id, set())
            if spec.requires_person and (
                person_category_id is None or person_category_id not in positive_categories
            ):
                skipped["activity_without_person"] += 1
                continue

            wrong_options = text_option_candidates(spec, positive_categories, rng)
            if len(wrong_options) < 3:
                skipped["not_enough_text_distractors"] += 1
                continue

            distractor_candidates = hard_distractor_candidates(
                image_id,
                target_category_id,
                positive_categories,
                set(spec.answer_category_ids),
                wrong_options,
                image_id_to_categories,
                images_by_category,
                category_id_to_name,
                category_id_to_supergroup,
                person_category_id,
                rng,
            )
            if len(distractor_candidates) < 3:
                skipped["not_enough_image_distractors"] += 1
                continue

            top_candidates = distractor_candidates[: min(TOP_K_DISTRACTORS, len(distractor_candidates))]
            distractors = rng.sample(top_candidates, 3)
            pool.append(
                {
                    "vqa_type": spec.vqa_type,
                    "question": spec.question,
                    "target_category": spec.target_category,
                    "answer_text": spec.answer_text,
                    "answer_category_names": list(spec.answer_category_names),
                    "relevant_image_id": image_id,
                    "relevant_categories": category_names_for_ids(
                        positive_categories, category_id_to_name
                    ),
                    "wrong_options": [
                        {
                            "text": option.text,
                            "category_names": list(option.category_names),
                        }
                        for option in wrong_options
                    ],
                    "distractors": distractors,
                }
            )

    rng.shuffle(pool)
    return pool, skipped


def select_blueprints(
    pool: list[dict[str, Any]],
    total_needed: int,
) -> list[dict[str, Any]]:
    selected = []
    used_relevant_images = set()

    for blueprint in pool:
        image_id = int(blueprint["relevant_image_id"])
        if image_id in used_relevant_images:
            continue
        selected.append(blueprint)
        used_relevant_images.add(image_id)
        if len(selected) == total_needed:
            return selected

    for blueprint in pool:
        if blueprint in selected:
            continue
        selected.append(blueprint)
        if len(selected) == total_needed:
            return selected

    fail(f"not enough usable RAG-VQA base samples: need {total_needed}, found {len(pool)}")


def make_text_options(
    answer_text: str,
    wrong_options: list[dict[str, Any]],
    answer_label: str,
) -> list[dict[str, str]]:
    wrong_iter = iter(wrong_options)
    text_options = []
    for label in TEXT_LABELS:
        if label == answer_label:
            text_options.append({"label": label, "text": answer_text})
        else:
            text_options.append({"label": label, "text": next(wrong_iter)["text"]})
    return text_options


def image_path(file_name: str, display_coco_root: str) -> str:
    return f"{display_coco_root}/val2017/{file_name}"


def make_image_record(
    position: int,
    image_id: int,
    image_id_to_file: dict[int, str],
    display_coco_root: str,
    is_relevant: bool,
    distractor_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    file_name = image_id_to_file[image_id]
    record: dict[str, Any] = {
        "position": position,
        "label": f"Image {position}",
        "image_id": image_id,
        "file_name": file_name,
        "path": image_path(file_name, display_coco_root),
        "is_relevant": is_relevant,
    }
    if not is_relevant and distractor_record is not None:
        record["shared_categories"] = list(distractor_record["shared_categories"])
        record["hard_negative_score"] = float(distractor_record["hard_negative_score"])
        record["category_overlap_score"] = float(distractor_record["category_overlap_score"])
        record["same_supergroup_bonus"] = float(distractor_record["same_supergroup_bonus"])
        record["person_cooccurrence_bonus"] = float(
            distractor_record["person_cooccurrence_bonus"]
        )
    return record


def balanced_image_layout(blueprint: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    relevant = {"image_id": int(blueprint["relevant_image_id"]), "is_relevant": True}
    d1, d2, d3 = [
        {"image_id": int(distractor["image_id"]), "is_relevant": False, "record": distractor}
        for distractor in blueprint["distractors"]
    ]
    return {
        1: [relevant, d1, d2, d3],
        2: [d3, relevant, d1, d2],
        3: [d2, d3, relevant, d1],
        4: [d1, d2, d3, relevant],
    }


def make_rows_for_split(
    split: str,
    blueprints: list[dict[str, Any]],
    image_id_to_file: dict[int, str],
    display_coco_root: str,
) -> list[dict[str, Any]]:
    rows = []
    for base_index, blueprint in enumerate(blueprints, start=1):
        answer_label = TEXT_LABELS[(base_index - 1) % len(TEXT_LABELS)]
        text_options = make_text_options(
            blueprint["answer_text"], blueprint["wrong_options"], answer_label
        )
        base_id = f"{split}_base_{base_index:06d}"
        layout = balanced_image_layout(blueprint)

        for relevant_position in IMAGE_POSITIONS:
            images = []
            for position, image_entry in zip(IMAGE_POSITIONS, layout[relevant_position]):
                images.append(
                    make_image_record(
                        position,
                        int(image_entry["image_id"]),
                        image_id_to_file,
                        display_coco_root,
                        bool(image_entry["is_relevant"]),
                        image_entry.get("record"),
                    )
                )

            rows.append(
                {
                    "sample_id": f"{split}_{base_index:06d}_imgpos_{relevant_position}",
                    "base_id": base_id,
                    "split": split,
                    "task_type": TASK_TYPE,
                    "vqa_type": blueprint["vqa_type"],
                    "question": blueprint["question"],
                    "text_options": text_options,
                    "answer": answer_label,
                    "answer_text": blueprint["answer_text"],
                    "target_category": blueprint["target_category"],
                    "answer_category_names": blueprint["answer_category_names"],
                    "relevant_image_id": int(blueprint["relevant_image_id"]),
                    "relevant_image_position": relevant_position,
                    "positive_position": f"Image {relevant_position}",
                    "relevant_categories": blueprint["relevant_categories"],
                    "images": images,
                    "options": copy.deepcopy(images),
                }
            )
    return rows


def group_by_base_id(rows: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row["base_id"]), []).append(row)
    return grouped


def validate_rows(rows: list[dict[str, Any]], project_root: Path) -> None:
    grouped = group_by_base_id(rows)
    answer_distribution = Counter(row_group[0]["answer"] for row_group in grouped.values())
    if answer_distribution:
        counts = [answer_distribution[label] for label in TEXT_LABELS]
        if max(counts) - min(counts) > 1:
            fail(f"answer labels are not balanced enough: {dict(answer_distribution)}")

    for base_id, base_rows in grouped.items():
        if len(base_rows) != 4:
            fail(f"{base_id} has {len(base_rows)} rows")
        if sorted(row["relevant_image_position"] for row in base_rows) != IMAGE_POSITIONS:
            fail(f"{base_id} relevant_image_position is not exactly 1/2/3/4")

        invariant_keys = ["question", "text_options", "answer", "answer_text"]
        reference = {key: base_rows[0][key] for key in invariant_keys}
        positions_by_image: dict[int, list[int]] = {}
        relevant_positions = []

        for row in base_rows:
            sample_id = row["sample_id"]
            for key, value in reference.items():
                if row[key] != value:
                    fail(f"{sample_id} changed base-level field: {key}")

            if row["answer"] not in TEXT_LABELS:
                fail(f"{sample_id} has invalid answer label: {row['answer']}")
            answer_matches = [
                option
                for option in row["text_options"]
                if option["label"] == row["answer"] and option["text"] == row["answer_text"]
            ]
            if len(answer_matches) != 1:
                fail(f"{sample_id} answer_text does not match text_options")

            if len(row["images"]) != 4:
                fail(f"{sample_id} does not have 4 images")
            if row["images"] != row["options"]:
                fail(f"{sample_id} images and options aliases differ")

            relevant_images = [image for image in row["images"] if image["is_relevant"]]
            if len(relevant_images) != 1:
                fail(f"{sample_id} has {len(relevant_images)} relevant images")

            relevant_image = relevant_images[0]
            if int(relevant_image["image_id"]) != int(row["relevant_image_id"]):
                fail(f"{sample_id} relevant_image_id does not match relevant image")
            if int(relevant_image["position"]) != int(row["relevant_image_position"]):
                fail(f"{sample_id} relevant_image_position does not match image position")
            if row["positive_position"] != f"Image {row['relevant_image_position']}":
                fail(f"{sample_id} positive_position is inconsistent")
            relevant_positions.append(int(relevant_image["position"]))

            image_ids = [int(image["image_id"]) for image in row["images"]]
            if len(image_ids) != len(set(image_ids)):
                fail(f"{sample_id} contains duplicate images")

            for image in row["images"]:
                path = Path(image["path"])
                if path.is_absolute():
                    fail(f"{sample_id} has absolute image path: {image['path']}")
                if not (project_root / path).is_file():
                    fail(f"{sample_id} missing image file: {image['path']}")
                positions_by_image.setdefault(int(image["image_id"]), []).append(
                    int(image["position"])
                )

        if sorted(relevant_positions) != IMAGE_POSITIONS:
            fail(f"{base_id} relevant image does not appear once at each position")
        if len(positions_by_image) != 4:
            fail(f"{base_id} does not contain exactly 4 unique images")
        for image_id, positions in positions_by_image.items():
            if sorted(positions) != IMAGE_POSITIONS:
                fail(f"{base_id}: image {image_id} positions are not balanced")


def verify_image_files(rows: list[dict[str, Any]], project_root: Path) -> None:
    missing = []
    for row in rows:
        for image in row["images"]:
            path = project_root / image["path"]
            if not path.is_file():
                missing.append(str(path))
                if len(missing) >= 10:
                    break
        if len(missing) >= 10:
            break
    if missing:
        fail(f"sampled image files are missing, first missing paths: {missing}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def average_negative_score(rows: list[dict[str, Any]]) -> float:
    scores = [
        float(image["hard_negative_score"])
        for row in rows
        for image in row["images"]
        if not image["is_relevant"]
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def answer_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped = group_by_base_id(rows)
    counter = Counter(row_group[0]["answer"] for row_group in grouped.values())
    return {label: counter.get(label, 0) for label in TEXT_LABELS}


def vqa_type_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped = group_by_base_id(rows)
    counter = Counter(row_group[0]["vqa_type"] for row_group in grouped.values())
    return dict(sorted(counter.items()))


def main() -> None:
    args = parse_args()
    if args.num_dev < 0 or args.num_test < 0:
        fail("num_dev and num_test must be non-negative")

    rng = random.Random(args.seed)
    coco_root = resolve_project_path(args.coco_root)
    caption_file = resolve_project_path(args.caption_file)
    instance_file = resolve_project_path(args.instance_file)
    out_dir = resolve_project_path(args.out_dir)
    image_root = require_inputs(coco_root, caption_file, instance_file)

    caption_coco = load_json(caption_file)
    instance_coco = load_json(instance_file)

    image_id_to_file = build_image_id_to_file(caption_coco)
    image_id_to_captions = build_image_id_to_captions(caption_coco)
    (
        category_id_to_name,
        name_to_category_id,
        category_id_to_supergroup,
        image_id_to_categories,
        images_by_category,
    ) = build_instance_indexes(instance_coco, image_id_to_file)

    task_specs = build_task_specs(name_to_category_id)
    if not task_specs:
        fail("no RAG-VQA task specs could be built from COCO categories")

    pool, skipped = build_candidate_pool(
        task_specs,
        image_id_to_file,
        image_id_to_categories,
        images_by_category,
        category_id_to_name,
        category_id_to_supergroup,
        name_to_category_id,
        rng,
    )
    total_needed = args.num_dev + args.num_test
    selected = select_blueprints(pool, total_needed)
    dev_blueprints = selected[: args.num_dev]
    test_blueprints = selected[args.num_dev :]

    display_coco_root = display_path(args.coco_root)
    dev_rows = make_rows_for_split(
        "rag_vqa_dev", dev_blueprints, image_id_to_file, display_coco_root
    )
    test_rows = make_rows_for_split(
        "rag_vqa_test", test_blueprints, image_id_to_file, display_coco_root
    )

    validate_rows(dev_rows, PROJECT_ROOT)
    validate_rows(test_rows, PROJECT_ROOT)
    verify_image_files(dev_rows + test_rows, PROJECT_ROOT)

    output_dev = out_dir / "orderguard_rag_vqa_dev.jsonl"
    output_test = out_dir / "orderguard_rag_vqa_test.jsonl"
    write_jsonl(output_dev, dev_rows)
    write_jsonl(output_test, test_rows)

    print(f"Usable base candidates: {len(pool)}")
    print(f"Captioned images indexed: {len(image_id_to_captions)}")
    print(f"Task specs: {len(task_specs)}")
    print(f"Dev base samples: {len(dev_blueprints)}")
    print(f"Test base samples: {len(test_blueprints)}")
    print(f"Dev rows: {len(dev_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Skipped candidate reasons: {dict(sorted(skipped.items()))}")
    print(f"Dev answer distribution: {answer_distribution(dev_rows)}")
    print(f"Test answer distribution: {answer_distribution(test_rows)}")
    print(f"Dev VQA type distribution: {vqa_type_distribution(dev_rows)}")
    print(f"Test VQA type distribution: {vqa_type_distribution(test_rows)}")
    print(f"Average hard negative score: {average_negative_score(dev_rows + test_rows):.4f}")
    print("All balance checks passed: true")
    print(f"Output dev: {display_path(output_dev)}")
    print(f"Output test: {display_path(output_test)}")


if __name__ == "__main__":
    main()
