# OrderGuard++ Methods

This note documents the aggregation methods implemented for the OrderGuard
experiments. The methods operate on saved Qwen output JSONL files, so they do
not require re-running the multimodal model.

## Input Assumption

Each raw output row contains one positioned version of a base sample:

- `base_id`: groups the four positioned versions.
- `prediction`: model output label, one of `A/B/C/D`.
- `options`: displayed image options with `label` and `image_id`.
- `answer_image_id`: gold image ID.
- `raw_response` or `evidence`: optional text evidence from the model.

The key step is mapping the predicted label back to the original `image_id`.
This avoids treating "A" as the same answer across different permutations.

## Method 1: Single Order Mean

Use every positioned sample as an ordinary one-shot evaluation case, then report
the accuracy over all rows.

```text
SingleOrderMean = correct positioned rows / all positioned rows
```

This is the fairest single-order baseline because it averages over correct
answers appearing at `A/B/C/D`. The script also prints `single_order_A` by
default as a position-specific diagnostic, but it should not be treated as the
main baseline because a model that favors A can look artificially strong when
the gold image is placed at A.

## Method 2: Permutation Voting

For each base sample, collect all positioned versions and map each predicted
label back to its original image ID.

```text
Vote(i) = sum_pi 1(model selects image i under permutation pi)
```

The final prediction is the image with the largest vote count.

This improves over single-order inference because the final decision is not
based on only one arbitrary ordering.

## Method 3: Position-Calibrated Voting

First estimate position bias from a dev set or from the evaluated output file:

```text
b(p) = count(model predicts position p) / count(valid predictions)
```

where `p` is one of `A/B/C/D`.

Then score each original image ID:

```text
S(i) = Vote(i) - lambda * sum_pi b(pos_pi(i))
```

where:

- `i` is an original image ID.
- `pi` is one positioned version.
- `pos_pi(i)` is the displayed label of image `i` in that version.
- `lambda` controls how strongly we subtract the position prior.

This transfers the paper's PIA intuition from evaluation to inference: if an
image receives support mainly because it appeared in positions the model often
chooses, its score is reduced.

## Method 4: Evidence-Aware OrderGuard++

The evidence-aware variant adds a consistency reward for repeated, similar
visual evidence attached to the same selected image.

```text
S(i) = Vote(i)
     - lambda * sum_pi b(pos_pi(i))
     + mu * EvidenceConsistency(i)
```

The current implementation extracts `Evidence: ...` from the raw response and
uses average pairwise token Jaccard similarity among evidence strings that vote
for the same image.

```text
EvidenceConsistency(i) =
  number_of_evidence_items(i) * average_pairwise_jaccard(evidence_i)
```

This is intentionally lightweight. It is robust enough for course experiments
and does not require an extra embedding model.

## Server Commands

Clean semantic-category hard negative setting:

```bash
cd /root/autodl-tmp/OrderGuard/NLP
PYTHONPATH=. python eval/run_orderguard_methods.py \
  --input_jsonl result/caption_semvis_hard_test_clean_qwen25vl.jsonl \
  --dev_jsonl result/caption_semvis_hard_dev_clean_qwen25vl.jsonl \
  --output_jsonl result/caption_semvis_hard_test_clean_orderguard_methods.jsonl \
  --metrics_json result/caption_semvis_hard_test_clean_orderguard_metrics.json \
  --lambda_bias 1.0 \
  --mu_evidence 0.5
```

If there is no dev output file yet, omit `--dev_jsonl`. The script will estimate
position bias from the input file itself:

```bash
PYTHONPATH=. python eval/run_orderguard_methods.py \
  --input_jsonl result/caption_semvis_hard_test_clean_qwen25vl.jsonl \
  --output_jsonl result/caption_semvis_hard_test_clean_orderguard_methods.jsonl \
  --metrics_json result/caption_semvis_hard_test_clean_orderguard_metrics.json
```

## Output

The script prints a base-level accuracy table:

```text
single_order_mean: accuracy=...
single_order_A: accuracy=...
permutation_voting: accuracy=...
position_calibrated: accuracy=...
evidence_orderguard: accuracy=...
```

The optional prediction JSONL contains one row per method per `base_id`, with:

- `method`
- `prediction_image_id`
- `answer_image_id`
- `is_correct`
- `scores`
- method-specific fields such as `position_bias`, `lambda_bias`, and
  `evidence_bonus`

## Reporting Interpretation

Use `single_order_mean` as the simple one-order baseline, `permutation_voting`
as the self-consistency baseline, and `position_calibrated` as the main proposed
method. Use `single_order_A/B/C/D` only for position-bias analysis.

Use `evidence_orderguard` as an enhanced variant. If it improves results, report
it as OrderGuard++. If it is close to position calibration, report it as an
analysis-oriented extension showing that evidence consistency can be integrated
without additional model calls.
