import json
import math
import re
from collections import Counter, defaultdict
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


def option_image_id(row, label):
    for option in row.get("options", []):
        if option.get("label") == label:
            return int(option["image_id"])
    return None


def option_label_for_image(row, image_id):
    target = int(image_id)
    for option in row.get("options", []):
        if int(option["image_id"]) == target:
            return option["label"]
    return None


def candidate_image_ids(group):
    ids = set()
    for row in group:
        for option in row.get("options", []):
            ids.add(int(option["image_id"]))
    return sorted(ids)


def prediction_image_id(row):
    pred = row.get("prediction")
    if pred not in POSITIONS:
        return None
    return option_image_id(row, pred)


def gold_image_id(group):
    return int(group[0]["answer_image_id"])


def extract_evidence(row):
    explicit = row.get("evidence")
    if explicit:
        return str(explicit).strip()

    raw = str(row.get("raw_response", "")).strip()
    if not raw:
        return ""

    match = re.search(
        r"Evidence\s*[:：]\s*(.+?)(?:\n\s*(?:Confidence|Answer)\s*[:：]|\Z)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return " ".join(match.group(1).strip().split())

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[1]
    return ""


def tokenize_evidence(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def jaccard_similarity(a, b):
    left = set(tokenize_evidence(a))
    right = set(tokenize_evidence(b))
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def pairwise_evidence_consistency(evidences):
    clean = [e for e in evidences if e]
    if len(clean) < 2:
        return 0.0

    scores = []
    for i, left in enumerate(clean):
        for right in clean[i + 1 :]:
            scores.append(jaccard_similarity(left, right))
    return sum(scores) / len(scores) if scores else 0.0


def majority_vote_scores(group):
    scores = Counter()
    evidence_by_image = defaultdict(list)

    for row in group:
        image_id = prediction_image_id(row)
        if image_id is None:
            continue
        scores[image_id] += 1.0
        evidence_by_image[image_id].append(extract_evidence(row))

    return scores, evidence_by_image


def choose_best(scores, tie_breaker_ids=None):
    if not scores:
        return None

    tie_breaker = {image_id: idx for idx, image_id in enumerate(tie_breaker_ids or [])}

    def sort_key(item):
        image_id, score = item
        return (-score, tie_breaker.get(image_id, math.inf), image_id)

    return sorted(scores.items(), key=sort_key)[0][0]


def summarize_base_predictions(records):
    if not records:
        return {
            "num_bases": 0,
            "accuracy": 0.0,
            "correct": 0,
        }

    correct = sum(1 for record in records if record["is_correct"])
    return {
        "num_bases": len(records),
        "accuracy": correct / len(records),
        "correct": correct,
    }

