#!/usr/bin/env bash
set -u

cd /root/autodl-tmp/OrderGuard/NLP

MODEL=/root/models/Qwen2.5-VL-7B-Instruct
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CORRUPTIONS=("blur" "noise" "occlusion")
KS=("4" "8" "12")
SEVERITY=2
MAX_NEW_TOKENS=32

echo "============================================================"
echo "RAG-VQA corruption permutation experiments"
echo "Model: ${MODEL}"
echo "Corruptions: ${CORRUPTIONS[*]}"
echo "K values: ${KS[*]}"
echo "Severity: ${SEVERITY}"
echo "============================================================"

for c in "${CORRUPTIONS[@]}"; do
  for k in "${KS[@]}"; do
    echo
    echo "============================================================"
    echo "Running RAG-VQA ${c} s${SEVERITY} perm${k}"
    echo "============================================================"

    DEV_OUT="result/rag_vqa_dev_${c}_s${SEVERITY}_perm${k}_qwen25vl.jsonl"
    TEST_OUT="result/rag_vqa_test_${c}_s${SEVERITY}_perm${k}_qwen25vl.jsonl"

    RAW_VOTE_OUT="result/rag_vqa_test_${c}_s${SEVERITY}_perm${k}_answer_voting_devweight_raw.jsonl"
    RAW_METRICS="result/rag_vqa_test_${c}_s${SEVERITY}_perm${k}_answer_voting_devweight_raw_metrics.json"

    SOFTMAX_VOTE_OUT="result/rag_vqa_test_${c}_s${SEVERITY}_perm${k}_answer_voting_devweight_softmax_t01.jsonl"
    SOFTMAX_METRICS="result/rag_vqa_test_${c}_s${SEVERITY}_perm${k}_answer_voting_devweight_softmax_t01_metrics.json"

    THRESHOLD_VOTE_OUT="result/rag_vqa_test_${c}_s${SEVERITY}_perm${k}_answer_voting_devweight_threshold_06.jsonl"
    THRESHOLD_METRICS="result/rag_vqa_test_${c}_s${SEVERITY}_perm${k}_answer_voting_devweight_threshold_06_metrics.json"

    echo "[1/5] Dev permutation -> ${DEV_OUT}"
    if [ -s "${DEV_OUT}" ]; then
      echo "Skip existing dev output: ${DEV_OUT}"
    else
      PYTHONPATH=. python eval/run_qwen_vl_rag_permutation_eval.py \
        --model_path "${MODEL}" \
        --input_jsonl data/orderguard_rag_vqa_dev.jsonl \
        --output_jsonl "${DEV_OUT}" \
        --project_root . \
        --corruption "${c}" \
        --severity "${SEVERITY}" \
        --num_permutations "${k}" \
        --max_new_tokens "${MAX_NEW_TOKENS}"
    fi

    echo "[2/5] Test permutation -> ${TEST_OUT}"
    if [ -s "${TEST_OUT}" ]; then
      echo "Skip existing test output: ${TEST_OUT}"
    else
      PYTHONPATH=. python eval/run_qwen_vl_rag_permutation_eval.py \
        --model_path "${MODEL}" \
        --input_jsonl data/orderguard_rag_vqa_test.jsonl \
        --output_jsonl "${TEST_OUT}" \
        --project_root . \
        --corruption "${c}" \
        --severity "${SEVERITY}" \
        --num_permutations "${k}" \
        --max_new_tokens "${MAX_NEW_TOKENS}"
    fi

    echo "[3/5] Raw position-weighted voting"
    PYTHONPATH=. python eval/run_rag_answer_voting.py \
      --input_jsonl "${TEST_OUT}" \
      --weight_jsonl "${DEV_OUT}" \
      --output_jsonl "${RAW_VOTE_OUT}" \
      --metrics_json "${RAW_METRICS}" \
      --weight_mode raw \
      --alpha 1.0 \
      --threshold 0.5 \
      --temperature 0.1

    echo "[4/5] Softmax position-weighted voting, T=0.1"
    PYTHONPATH=. python eval/run_rag_answer_voting.py \
      --input_jsonl "${TEST_OUT}" \
      --weight_jsonl "${DEV_OUT}" \
      --output_jsonl "${SOFTMAX_VOTE_OUT}" \
      --metrics_json "${SOFTMAX_METRICS}" \
      --weight_mode softmax \
      --temperature 0.1

    echo "[5/5] Threshold voting, threshold=0.6"
    PYTHONPATH=. python eval/run_rag_answer_voting.py \
      --input_jsonl "${TEST_OUT}" \
      --weight_jsonl "${DEV_OUT}" \
      --output_jsonl "${THRESHOLD_VOTE_OUT}" \
      --metrics_json "${THRESHOLD_METRICS}" \
      --weight_mode threshold \
      --threshold 0.6

    echo "Finished: ${c} s${SEVERITY} perm${k}"
  done
done

echo
echo "============================================================"
echo "All RAG-VQA corruption k=4/8/12 experiments finished."
echo "Metrics files:"
ls result/rag_vqa_test_*_s${SEVERITY}_perm*_answer_voting_devweight_*_metrics.json
echo "============================================================"
