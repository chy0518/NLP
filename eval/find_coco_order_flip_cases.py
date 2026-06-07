import argparse
import json
from collections import defaultdict
from pathlib import Path

POSITIONS = ("A", "B", "C", "D")


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


def group_by_base(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["base_id"]].append(row)
    return dict(grouped)


def method_records_by_base(rows):
    grouped = defaultdict(dict)
    for row in rows:
        method = row.get("method")
        if method and method != "single_order_mean":
            grouped[row["base_id"]][method] = row
    return dict(grouped)


def single_order_rows(rows, method_name):
    return [row for row in rows if row.get("method") == method_name]


def position_sort_key(row):
    position = row.get("positive_position")
    if position in POSITIONS:
        return POSITIONS.index(position)
    return 999


def choose_baseline_row(group, baseline_position):
    if baseline_position:
        for row in group:
            if row.get("positive_position") == baseline_position:
                return row
        return None
    return sorted(group, key=lambda row: row.get("sample_id", ""))[0]


def compact_order_record(row):
    return {
        "sample_id": row.get("sample_id"),
        "positive_position": row.get("positive_position"),
        "prediction_image_id": row.get("prediction_image_id"),
        "answer_image_id": row.get("answer_image_id"),
        "is_correct": bool(row.get("is_correct")),
        "scores": row.get("scores", {}),
    }


def compact_method_record(row):
    return {
        "method": row.get("method"),
        "prediction_image_id": row.get("prediction_image_id"),
        "answer_image_id": row.get("answer_image_id"),
        "is_correct": bool(row.get("is_correct")),
        "scores": row.get("scores", {}),
        "vote_scores": row.get("vote_scores", {}),
        "evidence_bonus": row.get("evidence_bonus", {}),
    }


def find_flip_cases(rows, baseline_position, method_name, require_all_other_wrong):
    singles = single_order_rows(rows, method_name)
    methods_by_base = method_records_by_base(rows)
    cases = []

    for base_id, group in sorted(group_by_base(singles).items()):
        group = sorted(group, key=position_sort_key)
        baseline = choose_baseline_row(group, baseline_position)
        if baseline is None or not baseline.get("is_correct"):
            continue

        changed_rows = [row for row in group if row is not baseline]
        wrong_after_reorder = [row for row in changed_rows if not row.get("is_correct")]
        if not wrong_after_reorder:
            continue
        if require_all_other_wrong and len(wrong_after_reorder) != len(changed_rows):
            continue

        flags = {
            row.get("positive_position", row.get("sample_id", "")): bool(row.get("is_correct"))
            for row in group
        }
        cases.append(
            {
                "base_id": base_id,
                "case_type": "baseline_correct_reordered_wrong",
                "baseline_position": baseline.get("positive_position"),
                "baseline_sample_id": baseline.get("sample_id"),
                "baseline_prediction_image_id": baseline.get("prediction_image_id"),
                "answer_image_id": baseline.get("answer_image_id"),
                "num_orders": len(group),
                "num_wrong_after_reorder": len(wrong_after_reorder),
                "position_correctness": flags,
                "wrong_positions_after_reorder": [
                    row.get("positive_position") for row in wrong_after_reorder
                ],
                "single_order_records": [compact_order_record(row) for row in group],
                "method_records": [
                    compact_method_record(row)
                    for _, row in sorted(methods_by_base.get(base_id, {}).items())
                ],
            }
        )

    return cases


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find COCO image-choice base samples where the baseline order is "
            "correct but at least one reordered version becomes wrong. The input "
            "is the JSONL produced by eval/run_orderguard_methods.py."
        )
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument(
        "--baseline_position",
        default="A",
        choices=["A", "B", "C", "D", "first"],
        help=(
            "Which positive_position is treated as the before-reordering case. "
            "Use 'first' to use the first sample_id in each base group."
        ),
    )
    parser.add_argument(
        "--method",
        default="single_order_mean",
        help="Single-order method rows to inspect in the OrderGuard output JSONL.",
    )
    parser.add_argument(
        "--require_all_other_wrong",
        action="store_true",
        help="Keep only cases where every non-baseline order is wrong.",
    )
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    baseline_position = None if args.baseline_position == "first" else args.baseline_position
    rows = load_jsonl(args.input_jsonl)
    cases = find_flip_cases(
        rows,
        baseline_position=baseline_position,
        method_name=args.method,
        require_all_other_wrong=args.require_all_other_wrong,
    )
    if args.top_k is not None:
        cases = cases[: args.top_k]

    write_jsonl(args.output_jsonl, cases)

    print("input:", args.input_jsonl)
    print("output:", args.output_jsonl)
    print("baseline_position:", args.baseline_position)
    print("method:", args.method)
    print("cases_found:", len(cases))
    if cases:
        print("first_case:", cases[0]["base_id"], cases[0]["position_correctness"])


if __name__ == "__main__":
    main()
