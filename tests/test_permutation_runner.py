from eval.run_qwen_vl_permutation_eval import (
    generate_permuted_samples,
    unique_base_samples,
)


def make_base_row(base_id="base_1"):
    return {
        "sample_id": f"{base_id}_pos_A",
        "base_id": base_id,
        "split": "test",
        "task_type": "caption_matching",
        "question": "Which image best matches the caption?",
        "answer": "A",
        "answer_image_id": 1,
        "positive_position": "A",
        "options": [
            {"label": "A", "image_id": 1, "path": "a.jpg", "is_correct": True},
            {"label": "B", "image_id": 2, "path": "b.jpg", "is_correct": False},
            {"label": "C", "image_id": 3, "path": "c.jpg", "is_correct": False},
            {"label": "D", "image_id": 4, "path": "d.jpg", "is_correct": False},
        ],
    }


def test_unique_base_samples_keeps_first_row_per_base():
    rows = [
        make_base_row("base_1"),
        {**make_base_row("base_1"), "sample_id": "base_1_pos_B"},
        make_base_row("base_2"),
    ]

    bases = unique_base_samples(rows)

    assert [row["base_id"] for row in bases] == ["base_1", "base_2"]
    assert [row["sample_id"] for row in bases] == ["base_1_pos_A", "base_2_pos_A"]


def test_generate_all_24_permutations_and_relabels_options():
    samples = generate_permuted_samples(
        make_base_row(),
        num_permutations=24,
        seed=42,
        corruption="clean",
        severity=1,
    )

    orders = {tuple(sample["permutation_image_ids"]) for sample in samples}

    assert len(samples) == 24
    assert len(orders) == 24
    assert all([opt["label"] for opt in sample["options"]] == ["A", "B", "C", "D"] for sample in samples)


def test_generate_sample_updates_answer_to_new_position():
    samples = generate_permuted_samples(
        make_base_row(),
        num_permutations=24,
        seed=42,
        corruption="clean",
        severity=1,
    )

    sample_with_answer_at_d = next(
        sample for sample in samples if sample["permutation_image_ids"][3] == 1
    )

    assert sample_with_answer_at_d["answer"] == "D"
    assert sample_with_answer_at_d["positive_position"] == "D"
    assert sample_with_answer_at_d["options"][3]["is_correct"] is True


def test_generate_sampled_permutations_is_reproducible():
    first = generate_permuted_samples(
        make_base_row(),
        num_permutations=12,
        seed=7,
        corruption="clean",
        severity=1,
    )
    second = generate_permuted_samples(
        make_base_row(),
        num_permutations=12,
        seed=7,
        corruption="clean",
        severity=1,
    )

    assert [x["permutation_image_ids"] for x in first] == [
        x["permutation_image_ids"] for x in second
    ]

