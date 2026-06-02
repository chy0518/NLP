import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path

ANSWER_LABELS = ("A", "B", "C", "D")


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


def choose_majority(counts, answer_labels=ANSWER_LABELS):
    if not counts:
        return None
    label_rank = {label: idx for idx, label in enumerate(answer_labels)}
    return sorted(
        counts.items(),
        key=lambda item: (-item[1], label_rank.get(item[0], len(answer_labels))),
    )[0][0]


def aggregate_base(group):
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


def aggregate_rows(rows):
    records = []
    for _, group in group_by_base_id(rows).items():
        records.append(aggregate_base(group))
    return records


def summarize(records):
    total = len(records)
    correct = sum(1 for record in records if record["is_correct"])
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
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--metrics_json", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    records = aggregate_rows(rows)
    metrics = summarize(records)

    print("input:", args.input_jsonl)
    print(
        "permutation_answer_voting: "
        f"accuracy={metrics['accuracy']:.4f} "
        f"correct={metrics['correct']}/{metrics['num_bases']}"
    )

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, records)
        print("wrote predictions:", args.output_jsonl)

    if args.metrics_json:
        out = {
            "input_jsonl": args.input_jsonl,
            "method": "permutation_answer_voting",
            **metrics,
        }
        with Path(args.metrics_json).open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("wrote metrics:", args.metrics_json)


if __name__ == "__main__":
    main()

