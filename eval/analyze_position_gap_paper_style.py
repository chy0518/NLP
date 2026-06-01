import argparse
import json
from collections import OrderedDict


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


def position_accuracy(rows, positions):
    result = {}
    for position in positions:
        position_rows = [row for row in rows if row.get("positive_position") == position]
        correct = sum(1 for row in position_rows if row.get("is_correct"))
        total = len(position_rows)
        acc = correct / total if total else 0.0
        result[position] = (correct, total, acc)
    return result


def position_gap(position_stats):
    accs = [acc for _, total, acc in position_stats.values() if total > 0]
    if not accs:
        return 0.0
    return max(accs) - min(accs)


def print_stats(title, stats):
    print(title)
    for position, (correct, total, acc) in stats.items():
        print(f"{position} {correct} / {total} = {acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    grouped = group_by_base_id(rows)
    positions = position_order(rows)

    all4_correct_bases = []
    unstable_bases = []
    for base_id, base_rows in grouped.items():
        if all(row.get("is_correct") for row in base_rows):
            all4_correct_bases.append(base_id)
        else:
            unstable_bases.append(base_id)

    unstable_rows = [
        row
        for base_id in unstable_bases
        for row in grouped[base_id]
    ]

    full_stats = position_accuracy(rows, positions)
    unstable_stats = position_accuracy(unstable_rows, positions)

    print(f"total_bases: {len(grouped)}")
    print(f"all4_correct_bases: {len(all4_correct_bases)}")
    print(f"unstable_bases: {len(unstable_bases)}")
    print_stats("Full-set position accuracy:", full_stats)
    print(f"full_position_gap: {position_gap(full_stats):.4f}")
    print_stats("Paper-style unstable-only position accuracy:", unstable_stats)
    print(f"unstable_position_gap: {position_gap(unstable_stats):.4f}")


if __name__ == "__main__":
    main()
