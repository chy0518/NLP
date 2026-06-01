import argparse
import json
from collections import Counter, OrderedDict


LEGACY_POSITIONS = ["A", "B", "C", "D"]
IMAGE_POSITIONS = ["Image 1", "Image 2", "Image 3", "Image 4"]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_by_base_id(rows):
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(row["base_id"], []).append(row)
    return grouped


def position_order(rows):
    positions = {row.get("positive_position") for row in rows}
    if positions and positions.issubset(set(IMAGE_POSITIONS)):
        return IMAGE_POSITIONS
    if positions and positions.issubset(set(LEGACY_POSITIONS)):
        return LEGACY_POSITIONS
    return sorted(position for position in positions if position is not None)


def accuracy(rows):
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("is_correct")) / len(rows)


def print_position_accuracy(rows):
    order = position_order(rows)
    for position in order:
        position_rows = [row for row in rows if row.get("positive_position") == position]
        correct = sum(1 for row in position_rows if row.get("is_correct"))
        total = len(position_rows)
        acc = correct / total if total else 0.0
        print(f"{position} {correct} / {total} = {acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    grouped = group_by_base_id(rows)
    prediction_frequency = Counter(row.get("prediction") for row in rows)

    all4_correct = 0
    any_correct = 0
    for base_rows in grouped.values():
        if all(row.get("is_correct") for row in base_rows):
            all4_correct += 1
        if any(row.get("is_correct") for row in base_rows):
            any_correct += 1

    print(f"n: {len(rows)}")
    print(f"accuracy: {accuracy(rows):.4f}")
    print(f"prediction_frequency: {dict(prediction_frequency)}")
    print("accuracy_by_positive_position:")
    print_position_accuracy(rows)
    print(f"num_bases: {len(grouped)}")
    if grouped:
        print(f"all_4_correct_consistency: {all4_correct / len(grouped):.4f}")
        print(f"any_correct: {any_correct / len(grouped):.4f}")
    else:
        print("all_4_correct_consistency: 0.0000")
        print("any_correct: 0.0000")


if __name__ == "__main__":
    main()
