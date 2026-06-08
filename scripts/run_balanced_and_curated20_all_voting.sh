#!/usr/bin/env bash
set -u

cd /root/autodl-tmp/OrderGuard/NLP

MODEL=/root/models/Qwen2.5-VL-7B-Instruct
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEVERITY=2

echo "============================================================"
echo "OrderGuard integrated voting experiments"
echo "Model: ${MODEL}"
echo "Start time: $(date)"
echo "============================================================"


# ============================================================
# Helper: image-choice position-weighted voting
# For Balanced Caption Matching
# prediction label A/B/C/D -> displayed option -> image_id
# Then aggregate votes over base_id.
# ============================================================

cat > scripts/run_image_choice_position_weighted_voting.py <<'PY'
import argparse
import json
import math
from collections import defaultdict, Counter
from pathlib import Path


LABELS = ["A", "B", "C", "D"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_position(row):
    pos = row.get("positive_position")
    if pos is None:
        pos = row.get("relevant_image_position")
    pos = str(pos)
    if pos in LABELS:
        return pos
    if pos.startswith("Image "):
        return pos
    return pos


def option_by_label(row, label):
    for opt in row.get("options", []):
        if opt.get("label") == label:
            return opt
    return None


def estimate_position_weights(rows):
    stats = {p: [0, 0] for p in LABELS}
    for r in rows:
        pos = get_position(r)
        pred = r.get("prediction")
        ans = r.get("answer")
        if pos not in stats:
            continue
        if pred not in LABELS:
            continue
        stats[pos][1] += 1
        if pred == ans:
            stats[pos][0] += 1

    weights = {}
    for p, (correct, total) in stats.items():
        weights[p] = correct / total if total else 0.0
    return weights, stats


def transform_weights(raw_weights, mode, alpha, threshold, temperature):
    if mode == "raw":
        return {p: w ** alpha for p, w in raw_weights.items()}

    if mode == "threshold":
        return {p: 1.0 if w >= threshold else 0.0 for p, w in raw_weights.items()}

    if mode == "softmax":
        mx = max(raw_weights.values()) if raw_weights else 0.0
        exps = {p: math.exp((w - mx) / temperature) for p, w in raw_weights.items()}
        s = sum(exps.values())
        return {p: (v / s if s else 0.0) for p, v in exps.items()}

    raise ValueError(f"Unknown weight_mode: {mode}")


def argmax_dict(d):
    if not d:
        return None
    return sorted(d.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--weight_jsonl", default=None)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--weight_mode", choices=["raw", "softmax", "threshold"], default="raw")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    rows = load_jsonl(args.input_jsonl)
    weight_rows = load_jsonl(args.weight_jsonl) if args.weight_jsonl else rows

    raw_weights, stats = estimate_position_weights(weight_rows)
    weights = transform_weights(raw_weights, args.weight_mode, args.alpha, args.threshold, args.temperature)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["base_id"]].append(r)

    preds = []
    single_correct = 0
    single_total = 0
    vote_correct = 0
    weighted_correct = 0

    for base_id, xs in sorted(grouped.items()):
        answer_image_id = xs[0].get("answer_image_id")

        raw_vote = Counter()
        weighted_scores = defaultdict(float)

        for r in xs:
            pred = r.get("prediction")
            if pred not in LABELS:
                continue

            opt = option_by_label(r, pred)
            if opt is None:
                continue

            pred_image_id = opt.get("image_id")
            raw_vote[pred_image_id] += 1

            # Use the display position of the predicted option, not gold position.
            pred_position = pred
            weighted_scores[pred_image_id] += weights.get(pred_position, 0.0)

            single_total += 1
            if pred_image_id == answer_image_id:
                single_correct += 1

        vote_pred = argmax_dict(raw_vote)
        weighted_pred = argmax_dict(weighted_scores)

        vote_ok = vote_pred == answer_image_id
        weighted_ok = weighted_pred == answer_image_id

        vote_correct += int(vote_ok)
        weighted_correct += int(weighted_ok)

        preds.append({
            "base_id": base_id,
            "answer_image_id": answer_image_id,
            "permutation_voting_prediction_image_id": vote_pred,
            "position_weighted_prediction_image_id": weighted_pred,
            "permutation_voting_correct": vote_ok,
            "position_weighted_correct": weighted_ok,
            "raw_vote": dict(raw_vote),
            "weighted_scores": dict(weighted_scores),
            "raw_position_weights": raw_weights,
            "position_weights": weights,
            "weight_mode": args.weight_mode,
            "alpha": args.alpha,
            "threshold": args.threshold,
            "temperature": args.temperature,
        })

    num_bases = len(grouped)
    metrics = {
        "input_jsonl": args.input_jsonl,
        "weight_jsonl": args.weight_jsonl,
        "weight_source": args.weight_jsonl or args.input_jsonl,
        "weight_mode": args.weight_mode,
        "alpha": args.alpha,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "raw_position_weights": raw_weights,
        "position_weights": weights,
        "position_weight_stats": {
            p: {
                "correct": stats[p][0],
                "total": stats[p][1],
                "accuracy": raw_weights[p],
            }
            for p in LABELS
        },
        "metrics": [
            {
                "method": "single_order_mean",
                "num_rows": single_total,
                "correct": single_correct,
                "accuracy": single_correct / single_total if single_total else 0.0,
            },
            {
                "method": "permutation_image_voting",
                "num_bases": num_bases,
                "correct": vote_correct,
                "accuracy": vote_correct / num_bases if num_bases else 0.0,
            },
            {
                "method": "position_weighted_image_voting",
                "num_bases": num_bases,
                "correct": weighted_correct,
                "accuracy": weighted_correct / num_bases if num_bases else 0.0,
            },
        ],
    }

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"input: {args.input_jsonl}")
    print(f"weight_source: {metrics['weight_source']}")
    print(f"weight_mode: {args.weight_mode} alpha={args.alpha} threshold={args.threshold} temperature={args.temperature}")
    print("position_weights:")
    for p in LABELS:
        st = stats[p]
        print(f"{p}: raw={raw_weights[p]:.4f} effective={weights[p]:.4f} correct={st[0]}/{st[1]}")
    for m in metrics["metrics"]:
        key = "num_bases" if "num_bases" in m else "num_rows"
        print(f"{m['method']}: accuracy={m['accuracy']:.4f} correct={m['correct']}/{m[key]}")
    print(f"wrote predictions: {args.output_jsonl}")
    print(f"wrote metrics: {args.metrics_json}")


if __name__ == "__main__":
    main()
PY


# ============================================================
# Part A. COCO Balanced 4-position Caption Matching
# Remaining corruptions: blur/noise
# Voting analysis for blur/noise/occlusion
# ============================================================

BALANCED_INPUT="data/orderguard_caption_semvis_hard_large_balanced_test.jsonl"
BALANCED_RUN_CORRUPTIONS=("blur" "noise")
BALANCED_VOTE_CORRUPTIONS=("blur" "noise" "occlusion")

echo ""
echo "============================================================"
echo "[Part A] COCO Balanced 4-position Caption Matching"
echo "============================================================"

for CORR in "${BALANCED_RUN_CORRUPTIONS[@]}"; do
  OUT="result/caption_semvis_hard_large_balanced_test_${CORR}_s${SEVERITY}_qwen25vl.jsonl"

  echo ""
  echo "------------------------------------------------------------"
  echo "[Balanced inference] corruption=${CORR}, severity=${SEVERITY}"
  echo "Output: ${OUT}"
  echo "------------------------------------------------------------"

  if [ -s "${OUT}" ]; then
    echo "[SKIP] ${OUT} already exists and is non-empty."
  else
    PYTHONPATH=. python eval/run_qwen_vl_eval.py \
      --model_path "${MODEL}" \
      --input_jsonl "${BALANCED_INPUT}" \
      --output_jsonl "${OUT}" \
      --project_root . \
      --corruption "${CORR}" \
      --severity "${SEVERITY}" \
      --max_new_tokens 64
  fi

  echo ""
  echo "[Balanced analyze_outputs] ${OUT}"
  PYTHONPATH=. python eval/analyze_outputs.py --input_jsonl "${OUT}"

  echo ""
  echo "[Balanced paper-style position gap] ${OUT}"
  PYTHONPATH=. python eval/analyze_position_gap_paper_style.py --input_jsonl "${OUT}"
done


echo ""
echo "============================================================"
echo "[Part A Voting] Balanced image-choice voting for blur/noise/occlusion"
echo "============================================================"

for CORR in "${BALANCED_VOTE_CORRUPTIONS[@]}"; do
  RAW_OUT="result/caption_semvis_hard_large_balanced_test_${CORR}_s${SEVERITY}_qwen25vl.jsonl"

  if [ ! -s "${RAW_OUT}" ]; then
    echo "[WARN] Missing ${RAW_OUT}; skip voting for ${CORR}."
    continue
  fi

  echo ""
  echo "------------------------------------------------------------"
  echo "[Balanced voting] corruption=${CORR}, severity=${SEVERITY}"
  echo "Input: ${RAW_OUT}"
  echo "------------------------------------------------------------"

  # raw alpha=1
  PYTHONPATH=. python scripts/run_image_choice_position_weighted_voting.py \
    --input_jsonl "${RAW_OUT}" \
    --output_jsonl "result/caption_semvis_hard_large_balanced_test_${CORR}_s${SEVERITY}_image_voting_raw.jsonl" \
    --metrics_json "result/caption_semvis_hard_large_balanced_test_${CORR}_s${SEVERITY}_image_voting_raw_metrics.json" \
    --weight_mode raw \
    --alpha 1.0

  cat "result/caption_semvis_hard_large_balanced_test_${CORR}_s${SEVERITY}_image_voting_raw_metrics.json"

  # softmax T=0.1
  PYTHONPATH=. python scripts/run_image_choice_position_weighted_voting.py \
    --input_jsonl "${RAW_OUT}" \
    --output_jsonl "result/caption_semvis_hard_large_balanced_test_${CORR}_s${SEVERITY}_image_voting_softmax_t01.jsonl" \
    --metrics_json "result/caption_semvis_hard_large_balanced_test_${CORR}_s${SEVERITY}_image_voting_softmax_t01_metrics.json" \
    --weight_mode softmax \
    --temperature 0.1

  cat "result/caption_semvis_hard_large_balanced_test_${CORR}_s${SEVERITY}_image_voting_softmax_t01_metrics.json"

  # threshold 0.6
  PYTHONPATH=. python scripts/run_image_choice_position_weighted_voting.py \
    --input_jsonl "${RAW_OUT}" \
    --output_jsonl "result/caption_semvis_hard_large_balanced_test_${CORR}_s${SEVERITY}_image_voting_threshold_06.jsonl" \
    --metrics_json "result/caption_semvis_hard_large_balanced_test_${CORR}_s${SEVERITY}_image_voting_threshold_06_metrics.json" \
    --weight_mode threshold \
    --threshold 0.6

  cat "result/caption_semvis_hard_large_balanced_test_${CORR}_s${SEVERITY}_image_voting_threshold_06_metrics.json"

done


# ============================================================
# Part B. Curated20 RAG-VQA
# blur/noise/occlusion x k=4/8/12
# Three text-answer voting modes
# ============================================================

CURATED_INPUT="data/orderguard_rag_vqa_retrieval_category_curated20.jsonl"
RAG_CORRUPTIONS=("blur" "noise" "occlusion")
KS=("4" "8" "12")
MAX_NEW_TOKENS=32

echo ""
echo "============================================================"
echo "[Part B] Curated20 RAG-VQA corruption permutation voting"
echo "============================================================"

for CORR in "${RAG_CORRUPTIONS[@]}"; do
  for K in "${KS[@]}"; do
    RAW_OUT="result/rag_vqa_retrieval_category_curated20_${CORR}_s${SEVERITY}_perm${K}_qwen25vl.jsonl"

    echo ""
    echo "------------------------------------------------------------"
    echo "[Curated20 RAG-VQA inference] corruption=${CORR}, severity=${SEVERITY}, k=${K}"
    echo "Output: ${RAW_OUT}"
    echo "------------------------------------------------------------"

    if [ -s "${RAW_OUT}" ]; then
      echo "[SKIP] ${RAW_OUT} already exists and is non-empty."
    else
      PYTHONPATH=. python eval/run_qwen_vl_rag_permutation_eval.py \
        --model_path "${MODEL}" \
        --input_jsonl "${CURATED_INPUT}" \
        --output_jsonl "${RAW_OUT}" \
        --project_root . \
        --corruption "${CORR}" \
        --severity "${SEVERITY}" \
        --num_permutations "${K}" \
        --max_new_tokens "${MAX_NEW_TOKENS}"
    fi

    # raw alpha=1
    PYTHONPATH=. python eval/run_rag_answer_voting.py \
      --input_jsonl "${RAW_OUT}" \
      --output_jsonl "result/rag_vqa_retrieval_category_curated20_${CORR}_s${SEVERITY}_perm${K}_answer_voting_raw.jsonl" \
      --metrics_json "result/rag_vqa_retrieval_category_curated20_${CORR}_s${SEVERITY}_perm${K}_answer_voting_raw_metrics.json" \
      --weight_mode raw \
      --alpha 1.0

    cat "result/rag_vqa_retrieval_category_curated20_${CORR}_s${SEVERITY}_perm${K}_answer_voting_raw_metrics.json"

    # softmax T=0.1
    PYTHONPATH=. python eval/run_rag_answer_voting.py \
      --input_jsonl "${RAW_OUT}" \
      --output_jsonl "result/rag_vqa_retrieval_category_curated20_${CORR}_s${SEVERITY}_perm${K}_answer_voting_softmax_t01.jsonl" \
      --metrics_json "result/rag_vqa_retrieval_category_curated20_${CORR}_s${SEVERITY}_perm${K}_answer_voting_softmax_t01_metrics.json" \
      --weight_mode softmax \
      --temperature 0.1

    cat "result/rag_vqa_retrieval_category_curated20_${CORR}_s${SEVERITY}_perm${K}_answer_voting_softmax_t01_metrics.json"

    # threshold 0.6
    PYTHONPATH=. python eval/run_rag_answer_voting.py \
      --input_jsonl "${RAW_OUT}" \
      --output_jsonl "result/rag_vqa_retrieval_category_curated20_${CORR}_s${SEVERITY}_perm${K}_answer_voting_threshold_06.jsonl" \
      --metrics_json "result/rag_vqa_retrieval_category_curated20_${CORR}_s${SEVERITY}_perm${K}_answer_voting_threshold_06_metrics.json" \
      --weight_mode threshold \
      --threshold 0.6

    cat "result/rag_vqa_retrieval_category_curated20_${CORR}_s${SEVERITY}_perm${K}_answer_voting_threshold_06_metrics.json"

  done
done


# ============================================================
# Final summary
# ============================================================

echo ""
echo "============================================================"
echo "[Final summary] Balanced image-choice voting metrics"
echo "============================================================"

python - <<'PY'
import json
from pathlib import Path

paths = sorted(Path("result").glob("caption_semvis_hard_large_balanced_test_*_s2_image_voting_*_metrics.json"))

print("| Corruption | Weight mode | Single-order mean | Image voting | Weighted image voting | Correct / Total |")
print("|---|---|---:|---:|---:|---:|")

for p in paths:
    name = p.name
    corruption = name.replace("caption_semvis_hard_large_balanced_test_", "").split("_s2_")[0]

    if "softmax_t01" in name:
        mode = "softmax T=0.1"
    elif "threshold_06" in name:
        mode = "threshold 0.6"
    elif "raw" in name:
        mode = "raw alpha=1"
    else:
        mode = "unknown"

    obj = json.loads(p.read_text())
    metrics = {m["method"]: m for m in obj.get("metrics", [])}

    sm = metrics.get("single_order_mean", {})
    pv = metrics.get("permutation_image_voting", {})
    wv = metrics.get("position_weighted_image_voting", {})

    sm_acc = sm.get("accuracy", 0)
    pv_acc = pv.get("accuracy", 0)
    wv_acc = wv.get("accuracy", 0)
    correct = wv.get("correct", "")
    total = wv.get("num_bases", "")

    print(f"| {corruption} | {mode} | {sm_acc*100:.2f}% | {pv_acc*100:.2f}% | {wv_acc*100:.2f}% | {correct} / {total} |")
PY


echo ""
echo "============================================================"
echo "[Final summary] Curated20 RAG-VQA voting metrics"
echo "============================================================"

python - <<'PY'
import json
from pathlib import Path

paths = sorted(Path("result").glob("rag_vqa_retrieval_category_curated20_*_s2_perm*_answer_voting_*_metrics.json"))

print("| Corruption | k | Weight mode | Permutation voting | Weighted voting | Correct / Total |")
print("|---|---:|---|---:|---:|---:|")

for p in paths:
    name = p.name
    stem = name.replace("rag_vqa_retrieval_category_curated20_", "")

    corruption = stem.split("_s2_")[0]
    k = stem.split("perm")[1].split("_")[0]

    if "softmax_t01" in name:
        mode = "softmax T=0.1"
    elif "threshold_06" in name:
        mode = "threshold 0.6"
    elif "raw" in name:
        mode = "raw alpha=1"
    else:
        mode = "unknown"

    obj = json.loads(p.read_text())
    metrics = {m["method"]: m for m in obj.get("metrics", [])}

    pv = metrics.get("permutation_answer_voting", {})
    wv = metrics.get("position_weighted_answer_voting", {})

    pv_acc = pv.get("accuracy", 0)
    wv_acc = wv.get("accuracy", 0)
    correct = wv.get("correct", "")
    total = wv.get("num_bases", "")

    print(f"| {corruption} | {k} | {mode} | {pv_acc*100:.2f}% | {wv_acc*100:.2f}% | {correct} / {total} |")
PY

echo ""
echo "============================================================"
echo "All experiments finished."
echo "End time: $(date)"
echo "============================================================"
