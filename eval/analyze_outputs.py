import argparse
import json
from collections import Counter, defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    args = parser.parse_args()

    rows = [json.loads(x) for x in open(args.input_jsonl, encoding="utf-8")]
    n = len(rows)

    if n == 0:
        print("empty file")
        return

    acc = sum(bool(r["is_correct"]) for r in rows) / n
    pred_counter = Counter(r.get("prediction") for r in rows)

    pos_total = Counter(r["positive_position"] for r in rows)
    pos_correct = Counter(r["positive_position"] for r in rows if r["is_correct"])

    print("file:", args.input_jsonl)
    print("n:", n)
    print("accuracy:", round(acc, 4))
    print("prediction_frequency:", dict(pred_counter))

    print("accuracy_by_positive_position:")
    for p in ["A", "B", "C", "D"]:
        total = pos_total[p]
        correct = pos_correct[p]
        print(p, correct, "/", total, round(correct / total, 4) if total else None)

    by_base = defaultdict(list)
    for r in rows:
        by_base[r["base_id"]].append(r)

    all4_correct = 0
    any_correct = 0

    for _, group in by_base.items():
        flags = [bool(x["is_correct"]) for x in group]

        if all(flags):
            all4_correct += 1
        if any(flags):
            any_correct += 1

    print("num_bases:", len(by_base))
    print("all_4_correct_consistency:", round(all4_correct / len(by_base), 4))
    print("any_correct:", round(any_correct / len(by_base), 4))


if __name__ == "__main__":
    main()
