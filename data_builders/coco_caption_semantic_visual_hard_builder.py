#!/usr/bin/env python3
"""Build COCO caption matching splits with semantic-visual hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
from pathlib import Path
from typing import Any, Iterable


OPTION_LABELS = ["A", "B", "C", "D"]
DEFAULT_SEED = 20260528
DEFAULT_TOP_K_HARD_NEGATIVES = 50
DEFAULT_ALPHA = 0.45
DEFAULT_BETA = 0.35
DEFAULT_GAMMA = 0.20
DEFAULT_CAPTION_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
TASK_TYPE = "caption_matching_semantic_visual_hard_negative"
DEFAULT_OUTPUT_PREFIX = "orderguard_caption_semvis_hard"
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
            "Construct 4-image COCO caption matching splits using hard negatives "
            "ranked by caption semantic similarity, optional CLIP image visual "
            "similarity, and COCO object-category overlap."
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
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--num_dev", type=int, default=30)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--top_k_hard_negatives",
        type=int,
        default=DEFAULT_TOP_K_HARD_NEGATIVES,
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--caption_model_name", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--clip_model_name", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--disable_image_similarity", action="store_true")
    parser.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
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


def stable_text_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_image_caption_indexes(
    caption_coco: dict[str, Any],
) -> tuple[dict[int, str], dict[int, list[str]], dict[int, str]]:
    image_id_to_file = {
        int(image["id"]): str(image["file_name"])
        for image in caption_coco.get("images", [])
    }
    if not image_id_to_file:
        fail("no images found in COCO caption annotations")

    image_id_to_captions: dict[int, list[str]] = {
        image_id: [] for image_id in image_id_to_file
    }
    for annotation in caption_coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in image_id_to_captions:
            continue

        caption = clean_caption(annotation.get("caption", ""))
        if caption:
            image_id_to_captions[image_id].append(caption)

    image_id_to_representative_caption = {
        image_id: " ".join(captions)
        for image_id, captions in image_id_to_captions.items()
        if captions
    }
    if not image_id_to_representative_caption:
        fail("no usable captions found in COCO caption annotations")

    return image_id_to_file, image_id_to_captions, image_id_to_representative_caption


def build_category_indexes(
    instance_coco: dict[str, Any],
    image_ids: Iterable[int],
) -> tuple[dict[int, str], dict[int, set[int]], dict[int, set[int]]]:
    category_id_to_name = {
        int(category["id"]): str(category["name"])
        for category in instance_coco.get("categories", [])
    }
    if not category_id_to_name:
        fail("no categories found in COCO instance annotations")

    image_id_to_categories = {int(image_id): set() for image_id in image_ids}
    category_id_to_image_ids: dict[int, set[int]] = {
        category_id: set() for category_id in category_id_to_name
    }

    for annotation in instance_coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in image_id_to_categories or category_id not in category_id_to_name:
            continue

        image_id_to_categories[image_id].add(category_id)
        category_id_to_image_ids.setdefault(category_id, set()).add(image_id)

    return category_id_to_name, image_id_to_categories, category_id_to_image_ids


def category_names_for_ids(
    category_ids: set[int],
    category_id_to_name: dict[int, str],
) -> list[str]:
    return [category_id_to_name[category_id] for category_id in sorted(category_ids)]


def jaccard_score(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def require_sentence_transformer_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm
    except ImportError as exc:
        fail(
            "caption semantic similarity requires sentence-transformers, numpy, "
            "and tqdm. Install with: pip install sentence-transformers tqdm numpy. "
            f"Original import error: {exc}"
        )

    return np, SentenceTransformer, tqdm


def load_or_compute_caption_embeddings(
    text_by_id: dict[str, str],
    model_name: str,
    cache_path: Path,
    batch_size: int = 128,
) -> dict[str, Any]:
    np, SentenceTransformer, tqdm = require_sentence_transformer_dependencies()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached: dict[str, Any] = {}
    if cache_path.is_file():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("model_name") == model_name:
            cached = payload.get("embeddings", {})

    missing_ids = [text_id for text_id in text_by_id if text_id not in cached]
    if missing_ids:
        try:
            model = SentenceTransformer(model_name)
        except Exception as exc:
            fail(
                "failed to load caption embedding model "
                f"'{model_name}'. If the model is not cached locally and the "
                "machine cannot access the network, download/cache it first. "
                f"Original error: {exc}"
            )

        for id_batch in tqdm(
            list(batched(missing_ids, batch_size)),
            desc="Encoding captions",
        ):
            texts = [text_by_id[text_id] for text_id in id_batch]
            embeddings = model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for text_id, embedding in zip(id_batch, embeddings):
                cached[text_id] = np.asarray(embedding, dtype=np.float32)

        with cache_path.open("wb") as f:
            pickle.dump(
                {
                    "model_name": model_name,
                    "embeddings": cached,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    return cached


def require_clip_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from PIL import Image
        from tqdm import tqdm
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        fail(
            "image visual similarity requires transformers, torch, pillow, numpy, "
            "and tqdm. Install with: pip install transformers pillow tqdm numpy "
            "torch, or rerun with --disable_image_similarity. "
            f"Original import error: {exc}"
        )

    return np, torch, Image, tqdm, (CLIPModel, CLIPProcessor)


def load_or_compute_image_embeddings(
    image_ids: list[int],
    image_id_to_file: dict[int, str],
    image_root: Path,
    model_name: str,
    cache_path: Path,
    batch_size: int = 32,
) -> dict[int, Any]:
    np, torch, Image, tqdm, clip_classes = require_clip_dependencies()
    CLIPModel, CLIPProcessor = clip_classes
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached: dict[int, Any] = {}
    if cache_path.is_file():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("model_name") == model_name:
            cached = payload.get("embeddings", {})

    missing_ids = [image_id for image_id in image_ids if image_id not in cached]
    if missing_ids:
        try:
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as exc:
            fail(
                "failed to load CLIP image model "
                f"'{model_name}'. If the model is not cached locally and the "
                "machine cannot access the network, download/cache it first, "
                "install the needed packages, or rerun with "
                f"--disable_image_similarity. Original error: {exc}"
            )

        device = "cpu"
        model.to(device)
        model.eval()

        for id_batch in tqdm(
            list(batched(missing_ids, batch_size)),
            desc="Encoding images",
        ):
            images = []
            for image_id in id_batch:
                image_path = image_root / image_id_to_file[image_id]
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))

            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            embeddings = features.detach().cpu().numpy()
            for image_id, embedding in zip(id_batch, embeddings):
                cached[int(image_id)] = np.asarray(embedding, dtype=np.float32)

        with cache_path.open("wb") as f:
            pickle.dump(
                {
                    "model_name": model_name,
                    "embeddings": cached,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    return cached


def cosine_similarity(left: Any, right: Any) -> float:
    return float(left @ right)


def collect_embedding_texts(
    image_id_to_representative_caption: dict[int, str],
    positive_records: list[dict[str, Any]],
) -> dict[str, str]:
    text_by_id = {}
    for image_id, caption in image_id_to_representative_caption.items():
        text_by_id[f"img:{image_id}"] = caption

    for positive in positive_records:
        caption = positive["caption"]
        text_by_id[f"caption:{stable_text_id(caption)}"] = caption

    return text_by_id


def build_positive_records(
    image_id_to_captions: dict[int, list[str]],
    image_id_to_categories: dict[int, set[int]],
    category_id_to_image_ids: dict[int, set[int]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    positives = []

    for image_id in sorted(image_id_to_captions):
        categories = image_id_to_categories.get(image_id, set())
        if len(categories) < 2:
            continue

        informative_captions = [
            caption
            for caption in image_id_to_captions[image_id]
            if is_informative_caption(caption)
        ]
        if not informative_captions:
            continue

        candidate_ids: set[int] = set()
        for category_id in categories:
            candidate_ids.update(category_id_to_image_ids.get(category_id, set()))
        candidate_ids.discard(image_id)
        if len(candidate_ids) < 3:
            continue

        positives.append(
            {
                "image_id": image_id,
                "caption": rng.choice(informative_captions),
            }
        )

    rng.shuffle(positives)
    return positives


def score_hard_negative_candidates(
    positive: dict[str, Any],
    image_id_to_categories: dict[int, set[int]],
    category_id_to_image_ids: dict[int, set[int]],
    category_id_to_name: dict[int, str],
    caption_embeddings: dict[str, Any],
    image_embeddings: dict[int, Any] | None,
    alpha: float,
    beta: float,
    gamma: float,
    top_k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    positive_image_id = int(positive["image_id"])
    positive_categories = image_id_to_categories[positive_image_id]
    positive_caption_id = f"caption:{stable_text_id(positive['caption'])}"
    positive_caption_embedding = caption_embeddings[positive_caption_id]
    positive_image_embedding = (
        image_embeddings[positive_image_id] if image_embeddings is not None else None
    )

    candidate_ids: set[int] = set()
    for category_id in positive_categories:
        candidate_ids.update(category_id_to_image_ids.get(category_id, set()))
    candidate_ids.discard(positive_image_id)

    candidates = []
    for candidate_image_id in candidate_ids:
        candidate_categories = image_id_to_categories.get(candidate_image_id, set())
        shared_category_ids = positive_categories & candidate_categories
        if not shared_category_ids:
            continue

        category_overlap_score = jaccard_score(positive_categories, candidate_categories)
        candidate_caption_id = f"img:{candidate_image_id}"
        if candidate_caption_id not in caption_embeddings:
            continue

        caption_similarity = cosine_similarity(
            positive_caption_embedding,
            caption_embeddings[candidate_caption_id],
        )

        if image_embeddings is None:
            image_similarity = 0.0
        else:
            image_similarity = cosine_similarity(
                positive_image_embedding,
                image_embeddings[candidate_image_id],
            )

        hard_score = (
            alpha * caption_similarity
            + beta * image_similarity
            + gamma * category_overlap_score
        )

        candidates.append(
            {
                "image_id": candidate_image_id,
                "hard_negative_score": hard_score,
                "caption_similarity": caption_similarity,
                "image_similarity": image_similarity,
                "category_overlap_score": category_overlap_score,
                "shared_categories": category_names_for_ids(
                    shared_category_ids, category_id_to_name
                ),
                "shared_count": len(shared_category_ids),
            }
        )

    if len(candidates) < 3:
        return []

    rng.shuffle(candidates)
    candidates.sort(
        key=lambda candidate: (
            candidate["hard_negative_score"],
            candidate["shared_count"],
            candidate["caption_similarity"],
            candidate["image_similarity"],
            candidate["category_overlap_score"],
        ),
        reverse=True,
    )
    return candidates[: min(top_k, len(candidates))]


def build_usable_positive_pool(
    positive_records: list[dict[str, Any]],
    image_id_to_categories: dict[int, set[int]],
    category_id_to_image_ids: dict[int, set[int]],
    category_id_to_name: dict[int, str],
    caption_embeddings: dict[str, Any],
    image_embeddings: dict[int, Any] | None,
    alpha: float,
    beta: float,
    gamma: float,
    top_k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    try:
        from tqdm import tqdm
    except ImportError as exc:
        fail(
            "building hard negative candidates requires tqdm for progress bars. "
            "Install with: pip install tqdm. "
            f"Original import error: {exc}"
        )

    usable = []
    for positive in tqdm(positive_records, desc="Scoring hard negatives"):
        image_id = int(positive["image_id"])
        category_ids = image_id_to_categories[image_id]
        hard_candidates = score_hard_negative_candidates(
            positive,
            image_id_to_categories,
            category_id_to_image_ids,
            category_id_to_name,
            caption_embeddings,
            image_embeddings,
            alpha,
            beta,
            gamma,
            top_k,
            rng,
        )
        if len(hard_candidates) < 3:
            continue

        usable.append(
            {
                "image_id": image_id,
                "caption": positive["caption"],
                "positive_categories": category_names_for_ids(
                    category_ids, category_id_to_name
                ),
                "hard_candidates": hard_candidates,
            }
        )

    rng.shuffle(usable)
    return usable


def relative_image_path(file_name: str, display_coco_root: str) -> str:
    return f"{display_coco_root}/val2017/{file_name}"


def image_option_record(
    label: str,
    image_id: int,
    image_id_to_file: dict[int, str],
    is_correct: bool,
    display_coco_root: str,
    negative: dict[str, Any] | None = None,
) -> dict[str, Any]:
    file_name = image_id_to_file[image_id]
    record: dict[str, Any] = {
        "label": label,
        "image_id": image_id,
        "file_name": file_name,
        "path": relative_image_path(file_name, display_coco_root),
        "is_correct": is_correct,
    }

    if negative is not None:
        record.update(
            {
                "hard_negative_score": round(float(negative["hard_negative_score"]), 4),
                "caption_similarity": round(float(negative["caption_similarity"]), 4),
                "image_similarity": round(float(negative["image_similarity"]), 4),
                "category_overlap_score": round(
                    float(negative["category_overlap_score"]), 4
                ),
                "shared_categories": list(negative["shared_categories"]),
            }
        )

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
    image_id_to_file: dict[int, str],
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
                        image_option_record(
                            label,
                            image_id,
                            image_id_to_file,
                            True,
                            display_coco_root,
                        )
                    )
                else:
                    negative = next(negative_iter)
                    image_id = int(negative["image_id"])
                    options.append(
                        image_option_record(
                            label,
                            image_id,
                            image_id_to_file,
                            False,
                            display_coco_root,
                            negative,
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


def validate_rows(rows: list[dict[str, Any]], image_root: Path) -> None:
    seen_sample_ids: set[str] = set()

    for row in rows:
        assert row["sample_id"] not in seen_sample_ids, row["sample_id"]
        seen_sample_ids.add(row["sample_id"])
        assert len(row["options"]) == 4, row["sample_id"]
        assert row["answer"] == row["positive_position"], row["sample_id"]
        assert row["task_type"] == TASK_TYPE, row["sample_id"]
        assert len(row["positive_categories"]) >= 2, row["sample_id"]

        correct_options = [option for option in row["options"] if option["is_correct"]]
        assert len(correct_options) == 1, row["sample_id"]
        assert correct_options[0]["label"] == row["answer"], row["sample_id"]
        assert int(correct_options[0]["image_id"]) == int(row["answer_image_id"]), (
            row["sample_id"]
        )

        option_image_ids = [int(option["image_id"]) for option in row["options"]]
        assert len(option_image_ids) == len(set(option_image_ids)), row["sample_id"]

        for option in row["options"]:
            assert (image_root / option["file_name"]).is_file(), option["path"]
            if option["is_correct"]:
                continue

            assert int(option["image_id"]) != int(row["answer_image_id"]), (
                row["sample_id"]
            )
            assert option.get("shared_categories"), row["sample_id"]
            assert option.get("hard_negative_score") is not None, row["sample_id"]
            assert option.get("caption_similarity") is not None, row["sample_id"]
            assert option.get("image_similarity") is not None, row["sample_id"]
            assert option.get("category_overlap_score") is not None, row["sample_id"]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def metric_averages(base_samples: list[dict[str, Any]]) -> dict[str, float]:
    negatives = [
        negative
        for base in base_samples
        for negative in base["negative_records"]
    ]
    if not negatives:
        return {
            "shared_categories": 0.0,
            "caption_similarity": 0.0,
            "image_similarity": 0.0,
            "category_overlap_score": 0.0,
            "hard_negative_score": 0.0,
        }

    return {
        "shared_categories": sum(len(n["shared_categories"]) for n in negatives)
        / len(negatives),
        "caption_similarity": sum(float(n["caption_similarity"]) for n in negatives)
        / len(negatives),
        "image_similarity": sum(float(n["image_similarity"]) for n in negatives)
        / len(negatives),
        "category_overlap_score": sum(float(n["category_overlap_score"]) for n in negatives)
        / len(negatives),
        "hard_negative_score": sum(float(n["hard_negative_score"]) for n in negatives)
        / len(negatives),
    }


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
    cache_dir = resolve_project_path(args.cache_dir)
    display_coco_root = display_path(args.coco_root)

    image_root = require_inputs(coco_root, caption_file, instance_file)

    rng = random.Random(args.seed)
    caption_coco = load_json(caption_file)
    instance_coco = load_json(instance_file)

    (
        image_id_to_file,
        image_id_to_captions,
        image_id_to_representative_caption,
    ) = build_image_caption_indexes(caption_coco)
    (
        category_id_to_name,
        image_id_to_categories,
        category_id_to_image_ids,
    ) = build_category_indexes(instance_coco, image_id_to_file.keys())

    positive_records = build_positive_records(
        image_id_to_captions,
        image_id_to_categories,
        category_id_to_image_ids,
        rng,
    )
    text_by_id = collect_embedding_texts(
        image_id_to_representative_caption,
        positive_records,
    )

    caption_embeddings = load_or_compute_caption_embeddings(
        text_by_id,
        args.caption_model_name,
        cache_dir / "caption_embeddings_semvis.pkl",
    )

    if args.disable_image_similarity:
        image_embeddings = None
    else:
        image_embeddings = load_or_compute_image_embeddings(
            sorted(image_id_to_file),
            image_id_to_file,
            image_root,
            args.clip_model_name,
            cache_dir / "image_embeddings_semvis.pkl",
        )

    usable_pool = build_usable_positive_pool(
        positive_records,
        image_id_to_categories,
        category_id_to_image_ids,
        category_id_to_name,
        caption_embeddings,
        image_embeddings,
        args.alpha,
        args.beta,
        args.gamma,
        args.top_k_hard_negatives,
        rng,
    )

    pool_offset = 0
    dev_bases, pool_offset = make_base_samples(
        "caption_semvis_hard_dev",
        args.num_dev,
        usable_pool,
        pool_offset,
        rng,
    )
    test_bases, pool_offset = make_base_samples(
        "caption_semvis_hard_test",
        args.num_test,
        usable_pool,
        pool_offset,
        rng,
    )

    dev_rows = expand_base_samples(dev_bases, image_id_to_file, display_coco_root)
    test_rows = expand_base_samples(test_bases, image_id_to_file, display_coco_root)
    validate_rows(dev_rows + test_rows, image_root)

    dev_path = out_dir / f"{args.output_prefix}_dev.jsonl"
    test_path = out_dir / f"{args.output_prefix}_test.jsonl"
    write_jsonl(dev_path, dev_rows)
    write_jsonl(test_path, test_rows)

    averages = metric_averages(dev_bases + test_bases)
    print(f"Usable positive samples: {len(usable_pool)}")
    print(f"Dev base samples: {len(dev_bases)}")
    print(f"Test base samples: {len(test_bases)}")
    print(f"Dev rows: {len(dev_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(
        "Average shared category count per negative: "
        f"{averages['shared_categories']:.2f}"
    )
    print(f"Average caption similarity: {averages['caption_similarity']:.4f}")
    print(f"Average image similarity: {averages['image_similarity']:.4f}")
    print(
        "Average category overlap score: "
        f"{averages['category_overlap_score']:.4f}"
    )
    print(
        "Average final hard negative score: "
        f"{averages['hard_negative_score']:.4f}"
    )
    print(f"Output dev: {relative_to_project(dev_path)}")
    print(f"Output test: {relative_to_project(test_path)}")
    print(f"Seed: {args.seed}")
    print(f"Top-K hard negatives: {args.top_k_hard_negatives}")
    print(
        "Image similarity: "
        f"{'disabled' if args.disable_image_similarity else args.clip_model_name}"
    )


if __name__ == "__main__":
    main()
