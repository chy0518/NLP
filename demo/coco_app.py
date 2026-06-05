import json
from collections import Counter, OrderedDict
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTION_LABELS = ("A", "B", "C", "D")

DEFAULT_PERMUTATION_PATH = "result/caption_semvis_hard_test_clean_perm8_qwen25vl.jsonl"
DEFAULT_METHODS_PATH = "result/caption_semvis_hard_test_clean_perm8_orderguard_methods.jsonl"


@st.cache_data(show_spinner=False)
def load_jsonl(path_text):
    path = resolve_path(path_text)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def group_by_base_id(rows):
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(row["base_id"], []).append(row)
    return grouped


def sorted_options(row):
    by_label = {option["label"]: option for option in row.get("options", [])}
    return [by_label[label] for label in OPTION_LABELS if label in by_label]


def option_image_id(row, label):
    for option in row.get("options", []):
        if option.get("label") == label:
            return int(option["image_id"])
    return None


def option_label_for_image(row, image_id):
    target = int(image_id)
    for option in row.get("options", []):
        if int(option["image_id"]) == target:
            return option.get("label")
    return None


def prediction_image_id(row):
    pred = row.get("prediction")
    if pred not in OPTION_LABELS:
        return None
    return option_image_id(row, pred)


def image_id_counts(rows):
    counts = Counter()
    for row in rows:
        image_id = prediction_image_id(row)
        if image_id is not None:
            counts[str(image_id)] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def label_counts(rows):
    counts = Counter()
    for row in rows:
        pred = row.get("prediction")
        if pred in OPTION_LABELS:
            counts[pred] += 1
    return {label: counts.get(label, 0) for label in OPTION_LABELS}


def majority_image_id(rows):
    counts = image_id_counts(rows)
    if not counts:
        return None
    return int(max(counts.items(), key=lambda item: (item[1], -int(item[0])))[0])


def correctness_badge(is_correct):
    return "correct" if is_correct else "wrong"


def image_path(option):
    return resolve_path(option["path"])


def selected_base_records(method_rows, base_id):
    records = {}
    for row in method_rows:
        if row.get("base_id") == base_id and row.get("method"):
            records[row["method"]] = row
    return records


def render_method_table(method_records):
    if not method_records:
        st.info("OrderGuard method output not loaded.")
        return

    rows = []
    preferred_order = [
        "permutation_voting",
        "position_calibrated",
        "evidence_orderguard",
        "single_order_mean",
        "single_order_A",
    ]
    ordered_names = [
        name for name in preferred_order if name in method_records
    ] + sorted(name for name in method_records if name not in preferred_order)

    for method in ordered_names:
        record = method_records[method]
        rows.append(
            {
                "method": method,
                "prediction_image_id": record.get("prediction_image_id"),
                "answer_image_id": record.get("answer_image_id"),
                "correct": correctness_badge(bool(record.get("is_correct"))),
            }
        )
    st.table(rows)


def render_summary(base_rows, method_records):
    first = base_rows[0]
    gold = int(first["answer_image_id"])
    majority = majority_image_id(base_rows)

    st.subheader("Sample Summary")
    st.write(f"**Base ID:** `{first['base_id']}`")
    if first.get("caption"):
        st.write(f"**Caption:** {first['caption']}")
    st.write(f"**Question:** {first.get('question', '')}")
    st.write(f"**Gold image ID:** `{gold}`")
    st.caption(
        "Predictions are option labels (A/B/C/D), but voting is computed after "
        "mapping each label back to the original COCO image_id."
    )

    cols = st.columns(4)
    cols[0].metric("Permutations", len(base_rows))
    cols[1].metric("Raw Majority Image", majority or "None")
    cols[2].metric("Majority Correct", correctness_badge(majority == gold))

    orderguard = (
        method_records.get("evidence_orderguard")
        or method_records.get("position_calibrated")
        or method_records.get("permutation_voting")
    )
    if orderguard:
        cols[3].metric(
            "Best Loaded Method",
            orderguard.get("prediction_image_id") or "None",
            correctness_badge(bool(orderguard.get("is_correct"))),
        )
    else:
        cols[3].metric("Best Loaded Method", "not loaded")

    st.subheader("OrderGuard Method Results")
    render_method_table(method_records)


def render_vote_charts(base_rows, method_records):
    st.subheader("Voting Analysis")
    col_left, col_right = st.columns(2)
    with col_left:
        st.caption("Raw prediction label counts")
        st.bar_chart(label_counts(base_rows))
    with col_right:
        st.caption("Raw vote counts after mapping labels to image_id")
        st.bar_chart(image_id_counts(base_rows))

    selected_method = st.selectbox(
        "Show method score details",
        ["none"] + sorted(method_records),
        index=0,
    )
    if selected_method != "none":
        record = method_records[selected_method]
        if record.get("scores"):
            st.caption(f"{selected_method} final scores")
            st.bar_chart(record["scores"])
        if record.get("vote_scores"):
            st.caption(f"{selected_method} raw vote scores")
            st.bar_chart(record["vote_scores"])
        if record.get("evidence_bonus"):
            st.caption(f"{selected_method} evidence bonus")
            st.bar_chart(record["evidence_bonus"])
        if record.get("position_bias"):
            st.caption(f"{selected_method} position bias")
            st.bar_chart(record["position_bias"])


def option_title(row, option):
    label = option.get("label")
    image_id = int(option["image_id"])
    parts = [f"Option {label}", f"ID {image_id}"]
    if image_id == int(row["answer_image_id"]):
        parts.append("gold")
    pred = row.get("prediction")
    if pred == label:
        parts.append("pred")
    return " | ".join(parts)


def render_permutation(row):
    pred_label = row.get("prediction") or "None"
    pred_image = prediction_image_id(row)
    answer_label = row.get("answer")
    answer_image = int(row["answer_image_id"])
    is_correct = bool(row.get("is_correct", pred_image == answer_image))
    title = (
        f"Permutation {row.get('permutation_index', '?')} | "
        f"Gold at {row.get('positive_position')} | "
        f"Pred label: {pred_label} | "
        f"Pred image: {pred_image or 'None'} | "
        f"{correctness_badge(is_correct)}"
    )

    with st.expander(title, expanded=False):
        st.write(f"**Predicted option label:** `{pred_label}`")
        st.write(f"**Predicted image ID:** `{pred_image}`")
        st.write(f"**Gold option label in this order:** `{answer_label}`")
        st.write(f"**Gold image ID:** `{answer_image}`")
        st.write(f"**Evidence:** {row.get('evidence') or row.get('raw_response', '')}")

        image_cols = st.columns(4)
        for col, option in zip(image_cols, sorted_options(row)):
            with col:
                st.markdown(f"**{option_title(row, option)}**")
                path = image_path(option)
                if path.is_file():
                    st.image(str(path), use_container_width=True)
                else:
                    st.warning(f"Missing image: {path}")


def render_permutations(base_rows):
    st.subheader("Permutation Details")
    sorted_rows = sorted(
        base_rows,
        key=lambda row: int(row.get("permutation_index", 0)),
    )
    for row in sorted_rows:
        render_permutation(row)


def main():
    st.set_page_config(page_title="OrderGuard COCO Demo", layout="wide")
    st.title("OrderGuard COCO Image-choice Demo")

    with st.sidebar:
        st.header("Data")
        permutation_path = st.text_input("Permutation JSONL", DEFAULT_PERMUTATION_PATH)
        methods_path = st.text_input("OrderGuard methods JSONL (optional)", DEFAULT_METHODS_PATH)
        load_methods = st.checkbox("Load method output", value=True)

    try:
        permutation_rows = load_jsonl(permutation_path)
    except Exception as exc:
        st.error(f"Failed to load permutation JSONL: {exc}")
        return

    grouped = group_by_base_id(permutation_rows)
    base_ids = list(grouped)
    if not base_ids:
        st.warning("No base samples found.")
        return

    method_rows = []
    if load_methods and methods_path.strip():
        try:
            method_rows = load_jsonl(methods_path)
        except Exception as exc:
            st.sidebar.warning(f"Method output not loaded: {exc}")

    with st.sidebar:
        st.header("Sample")
        base_id = st.selectbox("Base sample", base_ids)

    base_rows = grouped[base_id]
    method_records = selected_base_records(method_rows, base_id)

    render_summary(base_rows, method_records)
    render_vote_charts(base_rows, method_records)
    render_permutations(base_rows)


if __name__ == "__main__":
    main()
