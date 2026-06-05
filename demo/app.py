import json
from collections import Counter, OrderedDict
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANSWER_LABELS = ("A", "B", "C", "D")
IMAGE_POSITIONS = ("Image 1", "Image 2", "Image 3", "Image 4")

DEFAULT_PERMUTATION_PATH = "result/rag_vqa_test_clean_perm12_qwen25vl.jsonl"
DEFAULT_VOTING_PATH = "result/rag_vqa_test_clean_perm12_vote_softmax01.jsonl"


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


def sorted_images(row):
    return sorted(row.get("images", []), key=lambda image: int(image["position"]))


def sorted_text_options(row):
    by_label = {option["label"]: option for option in row.get("text_options", [])}
    return [by_label[label] for label in ANSWER_LABELS if label in by_label]


def answer_text_for_label(row, label):
    for option in row.get("text_options", []):
        if option.get("label") == label:
            return option.get("text", "")
    return ""


def prediction_counts(rows):
    counts = Counter()
    for row in rows:
        pred = row.get("prediction")
        if pred in ANSWER_LABELS:
            counts[pred] += 1
    return {label: counts.get(label, 0) for label in ANSWER_LABELS}


def majority_prediction(rows):
    counts = prediction_counts(rows)
    return max(ANSWER_LABELS, key=lambda label: (counts[label], -ANSWER_LABELS.index(label)))


def voting_records_by_method(rows):
    by_method = {}
    for row in rows:
        method = row.get("method")
        if method:
            by_method[method] = row
    return by_method


def selected_base_records(voting_rows, base_id):
    if not voting_rows:
        return {}
    records = [row for row in voting_rows if row.get("base_id") == base_id]
    return voting_records_by_method(records)


def image_path(image):
    return resolve_path(image["path"])


def correctness_badge(is_correct):
    return "correct" if is_correct else "wrong"


def render_text_options(row):
    st.subheader("Text Answer Options")
    option_cols = st.columns(4)
    answer = row.get("answer")
    for col, option in zip(option_cols, sorted_text_options(row)):
        label = option["label"]
        marker = "Gold" if label == answer else ""
        with col:
            st.metric(label=f"{label}. {option['text']}", value=marker or " ")


def render_summary(base_rows, voting_records):
    first = base_rows[0]
    counts = prediction_counts(base_rows)
    majority = majority_prediction(base_rows)

    st.subheader("Sample Summary")
    st.write(f"**Base ID:** `{first['base_id']}`")
    if first.get("retrieval_clue"):
        st.write(f"**Retrieval Clue:** {first['retrieval_clue']}")
    st.write(f"**Question:** {first.get('question', '')}")
    st.write(f"**Gold Answer:** `{first.get('answer')}` - {first.get('answer_text', '')}")
    st.caption(
        "Predictions are text answer labels (A/B/C/D). "
        "Relevant image position is shown separately as Image 1/2/3/4."
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Permutations", len(base_rows))
    summary_cols[1].metric("Raw Majority", majority)
    summary_cols[2].metric("Majority Correct", correctness_badge(majority == first.get("answer")))

    weighted = voting_records.get("position_weighted_answer_voting")
    if weighted:
        summary_cols[3].metric(
            "Weighted Prediction",
            weighted.get("prediction") or "None",
            correctness_badge(weighted.get("is_correct")),
        )
    else:
        summary_cols[3].metric("Weighted Prediction", "not loaded")

    st.caption(f"Raw vote counts: {counts}")
    render_text_options(first)


def render_vote_charts(base_rows, voting_records):
    st.subheader("Voting Analysis")
    counts = prediction_counts(base_rows)
    st.bar_chart(counts)

    weighted = voting_records.get("position_weighted_answer_voting")
    if weighted and weighted.get("weighted_scores"):
        st.caption("Weighted scores")
        st.bar_chart(weighted["weighted_scores"])

    if weighted and weighted.get("position_weights"):
        st.caption("Position weights used by weighted voting")
        st.bar_chart(weighted["position_weights"])


def render_permutation(row, display_index):
    pred = row.get("prediction") or "None"
    answer = row.get("answer")
    is_correct = bool(row.get("is_correct", pred == answer))
    position = row.get("positive_position")
    pred_text = answer_text_for_label(row, pred)
    answer_text = answer_text_for_label(row, answer) or row.get("answer_text", "")
    title = (
        f"Permutation {display_index} | "
        f"Relevant image: {position} | "
        f"Pred answer: {pred} | {correctness_badge(is_correct)}"
    )

    with st.expander(title, expanded=False):
        st.write(f"**Predicted text answer:** `{pred}` - {pred_text}")
        st.write(f"**Gold text answer:** `{answer}` - {answer_text}")
        st.write(f"**Evidence:** {row.get('evidence') or row.get('raw_response', '')}")
        image_cols = st.columns(4)
        for col, image in zip(image_cols, sorted_images(row)):
            with col:
                label = image.get("label", f"Image {image.get('position')}")
                if image.get("is_relevant"):
                    st.markdown(f"**{label} - relevant**")
                else:
                    st.markdown(f"**{label}**")
                path = image_path(image)
                if path.is_file():
                    st.image(str(path), use_container_width=True)
                else:
                    st.warning(f"Missing image: {path}")


def render_permutations(base_rows):
    st.subheader("Permutation Details")
    sorted_rows = sorted(
        enumerate(base_rows, start=1),
        key=lambda item: int(item[1].get("permutation_index", item[0])),
    )
    for fallback_index, row in sorted_rows:
        display_index = row.get("permutation_index", fallback_index)
        render_permutation(row, display_index)


def main():
    st.set_page_config(page_title="OrderGuard RAG-VQA Demo", layout="wide")
    st.title("OrderGuard RAG-VQA Permutation Demo")

    with st.sidebar:
        st.header("Data")
        permutation_path = st.text_input("Permutation JSONL", DEFAULT_PERMUTATION_PATH)
        voting_path = st.text_input("Voting JSONL (optional)", DEFAULT_VOTING_PATH)
        load_voting = st.checkbox("Load voting output", value=True)

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

    voting_rows = []
    if load_voting and voting_path.strip():
        try:
            voting_rows = load_jsonl(voting_path)
        except Exception as exc:
            st.sidebar.warning(f"Voting output not loaded: {exc}")

    with st.sidebar:
        st.header("Sample")
        base_id = st.selectbox("Base sample", base_ids)

    base_rows = grouped[base_id]
    voting_records = selected_base_records(voting_rows, base_id)

    render_summary(base_rows, voting_records)
    render_vote_charts(base_rows, voting_records)
    render_permutations(base_rows)


if __name__ == "__main__":
    main()
