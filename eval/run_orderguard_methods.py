import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from methods.aggregation import (
    load_jsonl,
    summarize_base_predictions,
    write_jsonl,
)
from methods.evidence_orderguard import aggregate_evidence_orderguard
from methods.position_calibration import (
    aggregate_permutation_voting,
    aggregate_position_calibrated,
    aggregate_single_order,
    estimate_position_bias,
)


def print_summary(name, records):
    summary = summarize_base_predictions(records)
    print(
        f"{name}: "
        f"accuracy={summary['accuracy']:.4f} "
        f"correct={summary['correct']}/{summary['num_bases']}"
    )
    return {"method": name, **summary}


def main():
    parser = argparse.ArgumentParser(
        description="Run OrderGuard aggregation methods on saved Qwen JSONL outputs."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument(
        "--dev_jsonl",
        default=None,
        help="Optional dev output JSONL for estimating position bias.",
    )
    parser.add_argument(
        "--output_jsonl",
        default=None,
        help="Optional path for base-level method predictions.",
    )
    parser.add_argument(
        "--metrics_json",
        default=None,
        help="Optional path for method summary metrics.",
    )
    parser.add_argument("--single_position", default="A", choices=["A", "B", "C", "D"])
    parser.add_argument("--lambda_bias", type=float, default=1.0)
    parser.add_argument("--mu_evidence", type=float, default=0.5)
    parser.add_argument("--bias_smoothing", type=float, default=1.0)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    bias_rows = load_jsonl(args.dev_jsonl) if args.dev_jsonl else rows
    position_bias = estimate_position_bias(
        bias_rows,
        smoothing=args.bias_smoothing,
    )

    print("input:", args.input_jsonl)
    if args.dev_jsonl:
        print("bias_source:", args.dev_jsonl)
    else:
        print("bias_source:", args.input_jsonl)
    print("position_bias:", json.dumps(position_bias, ensure_ascii=False))
    print()

    all_records = []
    metrics = []

    method_outputs = [
        (
            f"single_order_{args.single_position}",
            aggregate_single_order(rows, single_position=args.single_position),
        ),
        ("permutation_voting", aggregate_permutation_voting(rows)),
        (
            "position_calibrated",
            aggregate_position_calibrated(
                rows,
                position_bias=position_bias,
                lambda_bias=args.lambda_bias,
            ),
        ),
        (
            "evidence_orderguard",
            aggregate_evidence_orderguard(
                rows,
                position_bias=position_bias,
                lambda_bias=args.lambda_bias,
                mu_evidence=args.mu_evidence,
            ),
        ),
    ]

    for name, records in method_outputs:
        metrics.append(print_summary(name, records))
        all_records.extend(records)

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, all_records)
        print()
        print("wrote predictions:", args.output_jsonl)

    if args.metrics_json:
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_jsonl": args.input_jsonl,
                    "dev_jsonl": args.dev_jsonl,
                    "position_bias": position_bias,
                    "lambda_bias": args.lambda_bias,
                    "mu_evidence": args.mu_evidence,
                    "metrics": metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("wrote metrics:", args.metrics_json)


if __name__ == "__main__":
    main()
