import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path

ANSWER_LABELS = ("A", "B", "C", "D")
IMAGE_POSITIONS = ("Image 1", "Image 2", "Image 3", "Image 4")


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_by_base_id(rows):
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(row["base_id"], []).append(row)
    return grouped


def prediction_counts(rows):
    counts = Counter()
    invalid = 0
    for row in rows:
        pred = row.get("prediction")
        if pred in ANSWER_LABELS:
            counts[pred] += 1
        else:
            invalid += 1
    return counts, invalid


def relevant_position(row):
    position = row.get("positive_position")
    if position:
        return str(position)
    if row.get("relevant_image_position") is not None:
        return f"Image {row['relevant_image_position']}"
    return None


def estimate_position_weights(rows, smoothing=0.0):
    totals = Counter()
    correct = Counter()

    for row in rows:
        position = relevant_position(row)
        if position is None:
            continue

        totals[position] += 1
        if row.get("is_correct"):
            correct[position] += 1

    weights = {}
    for position in IMAGE_POSITIONS:
        denom = totals[position] + 2 * smoothing
        if denom:
            weights[position] = (correct[position] + smoothing) / denom
        else:
            weights[position] = 0.0

    return weights, {
        position: {
            "correct": correct[position],
            "total": totals[position],
            "accuracy": weights[position],
        }
        for position in IMAGE_POSITIONS
    }


def choose_majority(counts, answer_labels=ANSWER_LABELS):
    if not counts:
        return None
    label_rank = {label: idx for idx, label in enumerate(answer_labels)}
    return sorted(
        counts.items(),
        key=lambda item: (-item[1], label_rank.get(item[0], len(answer_labels))),
    )[0][0]


def choose_weighted(scores, answer_labels=ANSWER_LABELS):
    if not scores:
        return None
    label_rank = {label: idx for idx, label in enumerate(answer_labels)}
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], label_rank.get(item[0], len(answer_labels))),
    )[0][0]


def weighted_prediction_scores(rows, position_weights):
    scores = {label: 0.0 for label in ANSWER_LABELS}
    invalid = 0
    for row in rows:
        pred = row.get("prediction")
        if pred not in ANSWER_LABELS:
            invalid += 1
            continue

        position = relevant_position(row)
        scores[pred] += float(position_weights.get(position, 0.0))

    return scores, invalid


def aggregate_base_majority(group):
    first = group[0]
    counts, invalid = prediction_counts(group)
    prediction = choose_majority(counts)
    answer = first["answer"]
    return {
        "base_id": first["base_id"],
        "method": "permutation_answer_voting",
        "prediction": prediction,
        "answer": answer,
        "answer_text": first.get("answer_text"),
        "is_correct": prediction == answer,
        "num_votes": len(group),
        "num_valid_votes": sum(counts.values()),
        "num_invalid_votes": invalid,
        "vote_counts": {label: counts.get(label, 0) for label in ANSWER_LABELS},
        "question": first.get("question"),
        "text_options": first.get("text_options"),
    }


def aggregate_base_weighted(group, position_weights):
    first = group[0]
    scores, invalid = weighted_prediction_scores(group, position_weights)
    prediction = choose_weighted(scores)
    answer = first["answer"]
    return {
        "base_id": first["base_id"],
        "method": "position_weighted_answer_voting",
        "prediction": prediction,
        "answer": answer,
        "answer_text": first.get("answer_text"),
        "is_correct": prediction == answer,
        "num_votes": len(group),
        "num_valid_votes": len(group) - invalid,
        "num_invalid_votes": invalid,
        "weighted_scores": scores,
        "position_weights": position_weights,
        "question": first.get("question"),
        "text_options": first.get("text_options"),
    }


def aggregate_rows(rows, position_weights=None):
    records = []
    for _, group in group_by_base_id(rows).items():
        records.append(aggregate_base_majority(group))
        if position_weights is not None:
            records.append(aggregate_base_weighted(group, position_weights))
    return records


def summarize(records, method=None):
    selected = [
        record for record in records if method is None or record.get("method") == method
    ]
    total = len(selected)
    correct = sum(1 for record in selected if record["is_correct"])
    accuracy = correct / total if total else 0.0
    return {
        "num_bases": total,
        "correct": correct,
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate RAG-style VQA permutation outputs with answer voting."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument(
        "--weight_jsonl",
        default=None,
        help=(
            "Optional permutation output JSONL used only to estimate Image 1/2/3/4 "
            "reliability weights. If omitted, weights are estimated from input_jsonl."
        ),
    )
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--metrics_json", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    weight_rows = load_jsonl(args.weight_jsonl) if args.weight_jsonl else rows
    weight_source = args.weight_jsonl if args.weight_jsonl else args.input_jsonl
    position_weights, position_weight_stats = estimate_position_weights(weight_rows)
    records = aggregate_rows(rows, position_weights=position_weights)
    majority_metrics = summarize(records, method="permutation_answer_voting")
    weighted_metrics = summarize(records, method="position_weighted_answer_voting")

    print("input:", args.input_jsonl)
    print("weight_source:", weight_source)
    print("position_weights:")
    for position in IMAGE_POSITIONS:
        stats = position_weight_stats[position]
        print(
            f"{position}: weight={position_weights[position]:.4f} "
            f"correct={stats['correct']}/{stats['total']}"
        )
    print(
        "permutation_answer_voting: "
        f"accuracy={majority_metrics['accuracy']:.4f} "
        f"correct={majority_metrics['correct']}/{majority_metrics['num_bases']}"
    )
    print(
        "position_weighted_answer_voting: "
        f"accuracy={weighted_metrics['accuracy']:.4f} "
        f"correct={weighted_metrics['correct']}/{weighted_metrics['num_bases']}"
    )

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, records)
        print("wrote predictions:", args.output_jsonl)

    if args.metrics_json:
        out = {
            "input_jsonl": args.input_jsonl,
            "weight_jsonl": args.weight_jsonl,
            "weight_source": weight_source,
            "position_weights": position_weights,
            "position_weight_stats": position_weight_stats,
            "metrics": [
                {"method": "permutation_answer_voting", **majority_metrics},
                {"method": "position_weighted_answer_voting", **weighted_metrics},
            ],
        }
        with Path(args.metrics_json).open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("wrote metrics:", args.metrics_json)


if __name__ == "__main__":
    main()
