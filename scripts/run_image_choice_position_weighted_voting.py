import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ["A", "B", "C", "D"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_position(row):
    pos = row.get("positive_position")
    if pos is None:
        pos = row.get("relevant_image_position")
    pos = str(pos)
    if pos in LABELS:
        return pos
    if pos.startswith("Image "):
        return pos
    return pos


def option_by_label(row, label):
    for opt in row.get("options", []):
        if opt.get("label") == label:
            return opt
    return None


def estimate_position_weights(rows):
    stats = {position: [0, 0] for position in LABELS}
    for row in rows:
        position = get_position(row)
        prediction = row.get("prediction")
        answer = row.get("answer")
        if position not in stats:
            continue
        if prediction not in LABELS:
            continue
        stats[position][1] += 1
        if prediction == answer:
            stats[position][0] += 1

    weights = {}
    for position, (correct, total) in stats.items():
        weights[position] = correct / total if total else 0.0
    return weights, stats


def transform_weights(raw_weights, mode, alpha, threshold, temperature):
    if mode == "raw":
        return {
            position: weight**alpha
            for position, weight in raw_weights.items()
        }

    if mode == "threshold":
        return {
            position: 1.0 if weight >= threshold else 0.0
            for position, weight in raw_weights.items()
        }

    if mode == "softmax":
        mx = max(raw_weights.values()) if raw_weights else 0.0
        exp_values = {
            position: math.exp((weight - mx) / temperature)
            for position, weight in raw_weights.items()
        }
        denom = sum(exp_values.values())
        return {
            position: value / denom if denom else 0.0
            for position, value in exp_values.items()
        }

    raise ValueError(f"Unknown weight_mode: {mode}")


def argmax_dict(scores):
    if not scores:
        return None
    return sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--weight_jsonl", default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--metrics_json", required=True)
    parser.add_argument(
        "--weight_mode",
        choices=["raw", "softmax", "threshold"],
        default="raw",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    weight_rows = load_jsonl(args.weight_jsonl) if args.weight_jsonl else rows

    raw_weights, stats = estimate_position_weights(weight_rows)
    weights = transform_weights(
        raw_weights,
        args.weight_mode,
        args.alpha,
        args.threshold,
        args.temperature,
    )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["base_id"]].append(row)

    records = []
    single_correct = 0
    single_total = 0
    vote_correct = 0
    weighted_correct = 0

    for base_id, group in sorted(grouped.items()):
        answer_image_id = group[0].get("answer_image_id")

        raw_vote = Counter()
        weighted_scores = defaultdict(float)

        for row in group:
            prediction = row.get("prediction")
            if prediction not in LABELS:
                continue

            option = option_by_label(row, prediction)
            if option is None:
                continue

            pred_image_id = option.get("image_id")
            raw_vote[pred_image_id] += 1

            # Use the display position of the predicted option, not gold position.
            pred_position = prediction
            weighted_scores[pred_image_id] += weights.get(pred_position, 0.0)

            single_total += 1
            if pred_image_id == answer_image_id:
                single_correct += 1

        vote_pred = argmax_dict(raw_vote)
        weighted_pred = argmax_dict(weighted_scores)

        vote_ok = vote_pred == answer_image_id
        weighted_ok = weighted_pred == answer_image_id

        vote_correct += int(vote_ok)
        weighted_correct += int(weighted_ok)

        records.append(
            {
                "base_id": base_id,
                "answer_image_id": answer_image_id,
                "permutation_voting_prediction_image_id": vote_pred,
                "position_weighted_prediction_image_id": weighted_pred,
                "permutation_voting_correct": vote_ok,
                "position_weighted_correct": weighted_ok,
                "raw_vote": dict(raw_vote),
                "weighted_scores": dict(weighted_scores),
                "raw_position_weights": raw_weights,
                "position_weights": weights,
                "weight_mode": args.weight_mode,
                "alpha": args.alpha,
                "threshold": args.threshold,
                "temperature": args.temperature,
            }
        )

    num_bases = len(grouped)
    metrics = {
        "input_jsonl": args.input_jsonl,
        "weight_jsonl": args.weight_jsonl,
        "weight_source": args.weight_jsonl or args.input_jsonl,
        "weight_mode": args.weight_mode,
        "alpha": args.alpha,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "raw_position_weights": raw_weights,
        "position_weights": weights,
        "position_weight_stats": {
            p: {
                "correct": stats[p][0],
                "total": stats[p][1],
                "accuracy": raw_weights[p],
            }
            for p in LABELS
        },
        "metrics": [
            {
                "method": "single_order_mean",
                "num_rows": single_total,
                "correct": single_correct,
                "accuracy": single_correct / single_total if single_total else 0.0,
            },
            {
                "method": "permutation_image_voting",
                "num_bases": num_bases,
                "correct": vote_correct,
                "accuracy": vote_correct / num_bases if num_bases else 0.0,
            },
            {
                "method": "position_weighted_image_voting",
                "num_bases": num_bases,
                "correct": weighted_correct,
                "accuracy": weighted_correct / num_bases if num_bases else 0.0,
            },
        ],
    }

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"input: {args.input_jsonl}")
    print(f"weight_source: {metrics['weight_source']}")
    print(
        f"weight_mode: {args.weight_mode} "
        f"alpha={args.alpha} "
        f"threshold={args.threshold} "
        f"temperature={args.temperature}"
    )
    print("position_weights:")
    for position in LABELS:
        stat = stats[position]
        print(
            f"{position}: raw={raw_weights[position]:.4f} "
            f"effective={weights[position]:.4f} "
            f"correct={stat[0]}/{stat[1]}"
        )
    for metric in metrics["metrics"]:
        key = "num_bases" if "num_bases" in metric else "num_rows"
        print(
            f"{metric['method']}: accuracy={metric['accuracy']:.4f} "
            f"correct={metric['correct']}/{metric[key]}"
        )
    print(f"wrote predictions: {args.output_jsonl}")
    print(f"wrote metrics: {args.metrics_json}")


if __name__ == "__main__":
    main()
