import argparse
import json
from collections import defaultdict, Counter

POSITIONS = ["A", "B", "C", "D"]

def load_by_base(path):
    by_base = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_base[r["base_id"]].append(r)
    return by_base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    args = parser.parse_args()

    by_base = load_by_base(args.input_jsonl)

    total_bases = len(by_base)
    all4_correct_bases = 0
    unstable_bases = []

    all_pos_total = Counter()
    all_pos_correct = Counter()

    unstable_pos_total = Counter()
    unstable_pos_correct = Counter()

    for base_id, group in by_base.items():
        group = sorted(group, key=lambda x: x["positive_position"])
        flags = [bool(g["is_correct"]) for g in group]

        for g in group:
            pos = g["positive_position"]
            all_pos_total[pos] += 1
            if g["is_correct"]:
                all_pos_correct[pos] += 1

        if all(flags):
            all4_correct_bases += 1
            continue

        unstable_bases.append(base_id)
        for g in group:
            pos = g["positive_position"]
            unstable_pos_total[pos] += 1
            if g["is_correct"]:
                unstable_pos_correct[pos] += 1

    print("file:", args.input_jsonl)
    print("total_bases:", total_bases)
    print("all4_correct_bases:", all4_correct_bases)
    print("unstable_bases:", len(unstable_bases))
    print()

    print("Full-set position accuracy:")
    full_accs = []
    for p in POSITIONS:
        acc = all_pos_correct[p] / all_pos_total[p] if all_pos_total[p] else 0
        full_accs.append(acc)
        print(p, all_pos_correct[p], "/", all_pos_total[p], round(acc, 4))
    print("full_position_gap:", round(max(full_accs) - min(full_accs), 4))
    print()

    print("Paper-style unstable-only position accuracy:")
    unstable_accs = []
    for p in POSITIONS:
        acc = unstable_pos_correct[p] / unstable_pos_total[p] if unstable_pos_total[p] else 0
        unstable_accs.append(acc)
        print(p, unstable_pos_correct[p], "/", unstable_pos_total[p], round(acc, 4))
    print("unstable_position_gap:", round(max(unstable_accs) - min(unstable_accs), 4))

if __name__ == "__main__":
    main()
