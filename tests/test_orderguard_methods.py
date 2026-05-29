from methods.aggregation import (
    extract_evidence,
    option_image_id,
    prediction_image_id,
    summarize_base_predictions,
)
from methods.evidence_orderguard import aggregate_evidence_orderguard
from methods.position_calibration import (
    aggregate_permutation_voting,
    aggregate_position_calibrated,
    estimate_position_bias,
)


def make_row(base_id, positive_position, prediction, options):
    return {
        "sample_id": f"{base_id}_pos_{positive_position}",
        "base_id": base_id,
        "split": "test",
        "question": "Which image best matches the caption?",
        "answer": positive_position,
        "answer_image_id": 1,
        "positive_position": positive_position,
        "prediction": prediction,
        "is_correct": prediction == positive_position,
        "raw_response": (
            f"Answer: {prediction}\n"
            "Evidence: the image shows a person playing tennis on a court"
        ),
        "options": [
            {"label": label, "image_id": image_id, "is_correct": image_id == 1}
            for label, image_id in options
        ],
    }


def synthetic_rows():
    return [
        make_row("base_1", "A", "A", [("A", 1), ("B", 2), ("C", 3), ("D", 4)]),
        make_row("base_1", "B", "A", [("A", 2), ("B", 1), ("C", 3), ("D", 4)]),
        make_row("base_1", "C", "C", [("A", 2), ("B", 3), ("C", 1), ("D", 4)]),
        make_row("base_1", "D", "A", [("A", 2), ("B", 3), ("C", 4), ("D", 1)]),
    ]


def test_prediction_label_maps_to_image_id():
    row = synthetic_rows()[1]

    assert option_image_id(row, "A") == 2
    assert option_image_id(row, "B") == 1
    assert prediction_image_id(row) == 2


def test_estimate_position_bias_counts_predictions():
    bias = estimate_position_bias(synthetic_rows(), smoothing=0.0)

    assert bias == {"A": 0.75, "B": 0.0, "C": 0.25, "D": 0.0}


def test_permutation_voting_aggregates_by_original_image_id():
    records = aggregate_permutation_voting(synthetic_rows())

    assert len(records) == 1
    assert records[0]["scores"] == {"1": 2.0, "2": 2.0}
    assert records[0]["prediction_image_id"] == 1
    assert records[0]["is_correct"] is True


def test_position_calibration_penalizes_overused_positions():
    rows = synthetic_rows()
    bias = estimate_position_bias(rows, smoothing=0.0)
    records = aggregate_position_calibrated(rows, bias, lambda_bias=1.0)

    assert len(records) == 1
    assert records[0]["prediction_image_id"] == 1
    assert records[0]["scores"]["1"] > records[0]["scores"]["2"]


def test_evidence_orderguard_keeps_base_level_metrics():
    rows = synthetic_rows()
    bias = estimate_position_bias(rows, smoothing=0.0)
    records = aggregate_evidence_orderguard(
        rows,
        position_bias=bias,
        lambda_bias=1.0,
        mu_evidence=0.5,
    )
    summary = summarize_base_predictions(records)

    assert records[0]["prediction_image_id"] == 1
    assert summary["accuracy"] == 1.0
    assert summary["correct"] == 1


def test_extract_evidence_from_raw_response():
    row = {
        "raw_response": "Answer: B\nEvidence: a red bus is parked on the street\nConfidence: 0.8"
    }

    assert extract_evidence(row) == "a red bus is parked on the street"

