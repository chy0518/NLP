import argparse
import json
from collections import defaultdict

def load(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    by_base = defaultdict(list)
    for r in rows:
        by_base[r["base_id"]].append(r)
    return by_base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    by_base = load(args.input_jsonl)
    found = 0

    for base_id, group in by_base.items():
        group = sorted(group, key=lambda x: x["positive_position"])
        flags = [g["is_correct"] for g in group]

        if any(flags) and not all(flags):
            print("=" * 100)
            print("base_id:", base_id)
            print("question:", group[0]["question"])
            print("answer_image_id:", group[0]["answer_image_id"])
            print("flags:", list(zip([g["positive_position"] for g in group], flags)))
            for g in group:
                print("-" * 60)
                print("sample_id:", g["sample_id"])
                print("gold:", g["answer"], "pred:", g["prediction"], "correct:", g["is_correct"])
                print("response:", g["raw_response"].replace("\n", " ")[:300])
                print("options:")
                for opt in g["options"]:
                    print(opt["label"], opt["image_id"], opt["path"], "correct=", opt["is_correct"])
            found += 1
            if found >= args.top_k:
                break

    print("found:", found)

if __name__ == "__main__":
    main()
