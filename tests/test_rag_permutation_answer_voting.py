from eval.run_qwen_vl_rag_permutation_eval import (
    generate_permuted_samples,
    unique_base_samples,
)
from eval.run_rag_answer_voting import aggregate_rows, summarize


def make_rag_base(base_id="rag_base_1"):
    return {
        "sample_id": f"{base_id}_imgpos_1",
        "base_id": base_id,
        "split": "rag_vqa_test",
        "task_type": "rag_style_multi_image_vqa",
        "vqa_type": "object_existence",
        "question": "Which object is present in the relevant image?",
        "text_options": [
            {"label": "A", "text": "pizza"},
            {"label": "B", "text": "broccoli"},
            {"label": "C", "text": "donut"},
            {"label": "D", "text": "banana"},
        ],
        "answer": "A",
        "answer_text": "pizza",
        "target_category": "pizza",
        "relevant_image_id": 1,
        "relevant_image_position": 1,
        "positive_position": "Image 1",
        "images": [
            {"position": 1, "label": "Image 1", "image_id": 1, "path": "1.jpg", "is_relevant": True},
            {"position": 2, "label": "Image 2", "image_id": 2, "path": "2.jpg", "is_relevant": False},
            {"position": 3, "label": "Image 3", "image_id": 3, "path": "3.jpg", "is_relevant": False},
            {"position": 4, "label": "Image 4", "image_id": 4, "path": "4.jpg", "is_relevant": False},
        ],
    }


def test_rag_unique_base_samples_rejects_non_rag_rows():
    rows = [
        make_rag_base("base_1"),
        {**make_rag_base("base_1"), "sample_id": "base_1_imgpos_2"},
        make_rag_base("base_2"),
    ]

    bases = unique_base_samples(rows)

    assert [row["base_id"] for row in bases] == ["base_1", "base_2"]


def test_rag_permutations_keep_text_answer_and_relabel_images():
    samples = generate_permuted_samples(
        make_rag_base(),
        num_permutations=24,
        seed=42,
        corruption="clean",
        severity=1,
    )
    orders = {tuple(sample["permutation_image_ids"]) for sample in samples}

    assert len(samples) == 24
    assert len(orders) == 24
    assert all(sample["answer"] == "A" for sample in samples)
    assert all(sample["answer_text"] == "pizza" for sample in samples)
    assert all([image["position"] for image in sample["images"]] == [1, 2, 3, 4] for sample in samples)


def test_rag_permutation_updates_relevant_image_position():
    samples = generate_permuted_samples(
        make_rag_base(),
        num_permutations=24,
        seed=42,
        corruption="clean",
        severity=1,
    )
    sample = next(sample for sample in samples if sample["permutation_image_ids"][3] == 1)

    assert sample["relevant_image_position"] == 4
    assert sample["positive_position"] == "Image 4"
    assert sample["images"][3]["is_relevant"] is True


def test_rag_answer_voting_aggregates_text_predictions_by_base():
    base = make_rag_base()
    rows = []
    for idx, pred in enumerate(["A", "B", "A", "C"]):
        row = dict(base)
        row["sample_id"] = f"base_perm_{idx}"
        row["prediction"] = pred
        row["is_correct"] = pred == "A"
        rows.append(row)

    records = aggregate_rows(rows)
    metrics = summarize(records)

    assert records[0]["prediction"] == "A"
    assert records[0]["vote_counts"] == {"A": 2, "B": 1, "C": 1, "D": 0}
    assert metrics["accuracy"] == 1.0


def test_rag_answer_voting_tie_breaks_by_answer_label_order():
    base = make_rag_base()
    rows = []
    for idx, pred in enumerate(["B", "A"]):
        row = dict(base)
        row["sample_id"] = f"base_perm_{idx}"
        row["prediction"] = pred
        row["is_correct"] = pred == "A"
        rows.append(row)

    records = aggregate_rows(rows)

    assert records[0]["prediction"] == "A"

