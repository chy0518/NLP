from methods.aggregation import (
    candidate_image_ids,
    choose_best,
    gold_image_id,
    group_by_base,
    majority_vote_scores,
    pairwise_evidence_consistency,
)
from methods.position_calibration import position_penalty_for_image


def evidence_bonus_by_image(evidence_by_image):
    bonuses = {}
    for image_id, evidences in evidence_by_image.items():
        consistency = pairwise_evidence_consistency(evidences)
        bonuses[image_id] = len(evidences) * consistency
    return bonuses


def aggregate_evidence_orderguard(
    rows,
    position_bias,
    lambda_bias=1.0,
    mu_evidence=0.5,
):
    records = []
    for base_id, group in sorted(group_by_base(rows).items()):
        vote_scores, evidence_by_image = majority_vote_scores(group)
        evidence_bonus = evidence_bonus_by_image(evidence_by_image)
        scores = {}

        for image_id in candidate_image_ids(group):
            vote = float(vote_scores.get(image_id, 0.0))
            penalty = position_penalty_for_image(group, image_id, position_bias)
            bonus = float(evidence_bonus.get(image_id, 0.0))
            scores[image_id] = vote - lambda_bias * penalty + mu_evidence * bonus

        tie_breaker = candidate_image_ids(group)
        pred_image_id = choose_best(scores, tie_breaker)
        gold = gold_image_id(group)
        records.append(
            {
                "base_id": base_id,
                "method": "evidence_orderguard",
                "prediction_image_id": pred_image_id,
                "answer_image_id": gold,
                "is_correct": pred_image_id == gold,
                "lambda_bias": lambda_bias,
                "mu_evidence": mu_evidence,
                "position_bias": position_bias,
                "scores": {str(k): v for k, v in scores.items()},
                "vote_scores": {str(k): v for k, v in vote_scores.items()},
                "evidence_bonus": {str(k): v for k, v in evidence_bonus.items()},
            }
        )
    return records

