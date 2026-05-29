from collections import Counter

from methods.aggregation import (
    POSITIONS,
    candidate_image_ids,
    choose_best,
    gold_image_id,
    group_by_base,
    majority_vote_scores,
    option_label_for_image,
)


def estimate_position_bias(rows, smoothing=1.0):
    counts = Counter()
    total = 0.0

    for row in rows:
        pred = row.get("prediction")
        if pred in POSITIONS:
            counts[pred] += 1.0
            total += 1.0

    denom = total + smoothing * len(POSITIONS)
    if denom == 0:
        return {position: 1.0 / len(POSITIONS) for position in POSITIONS}

    return {
        position: (counts[position] + smoothing) / denom
        for position in POSITIONS
    }


def aggregate_single_order(rows, single_position="A"):
    records = []
    for base_id, group in sorted(group_by_base(rows).items()):
        selected_row = next(
            (row for row in group if row.get("positive_position") == single_position),
            sorted(group, key=lambda row: row["sample_id"])[0],
        )
        scores, _ = majority_vote_scores([selected_row])
        tie_breaker = candidate_image_ids(group)
        pred_image_id = choose_best(scores, tie_breaker)
        gold = gold_image_id(group)
        records.append(
            {
                "base_id": base_id,
                "method": f"single_order_{single_position}",
                "prediction_image_id": pred_image_id,
                "answer_image_id": gold,
                "is_correct": pred_image_id == gold,
                "scores": {str(k): v for k, v in scores.items()},
            }
        )
    return records


def aggregate_permutation_voting(rows):
    records = []
    for base_id, group in sorted(group_by_base(rows).items()):
        scores, evidence_by_image = majority_vote_scores(group)
        tie_breaker = candidate_image_ids(group)
        pred_image_id = choose_best(scores, tie_breaker)
        gold = gold_image_id(group)
        records.append(
            {
                "base_id": base_id,
                "method": "permutation_voting",
                "prediction_image_id": pred_image_id,
                "answer_image_id": gold,
                "is_correct": pred_image_id == gold,
                "scores": {str(k): v for k, v in scores.items()},
                "num_evidence_items": {
                    str(k): len(v) for k, v in evidence_by_image.items()
                },
            }
        )
    return records


def position_penalty_for_image(group, image_id, position_bias):
    penalty = 0.0
    for row in group:
        label = option_label_for_image(row, image_id)
        if label in POSITIONS:
            penalty += position_bias[label]
    return penalty


def aggregate_position_calibrated(rows, position_bias, lambda_bias=1.0):
    records = []
    for base_id, group in sorted(group_by_base(rows).items()):
        vote_scores, _ = majority_vote_scores(group)
        scores = {}

        for image_id in candidate_image_ids(group):
            vote = float(vote_scores.get(image_id, 0.0))
            penalty = position_penalty_for_image(group, image_id, position_bias)
            scores[image_id] = vote - lambda_bias * penalty

        tie_breaker = candidate_image_ids(group)
        pred_image_id = choose_best(scores, tie_breaker)
        gold = gold_image_id(group)
        records.append(
            {
                "base_id": base_id,
                "method": "position_calibrated",
                "prediction_image_id": pred_image_id,
                "answer_image_id": gold,
                "is_correct": pred_image_id == gold,
                "lambda_bias": lambda_bias,
                "position_bias": position_bias,
                "scores": {str(k): v for k, v in scores.items()},
                "vote_scores": {str(k): v for k, v in vote_scores.items()},
            }
        )
    return records

