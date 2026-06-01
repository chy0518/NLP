import argparse
import copy
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from methods.corruption import corrupt_option_images
except ImportError:
    corrupt_option_images = None


RAG_TASK_TYPE = "rag_style_multi_image_vqa"
ANSWER_LABELS = ["A", "B", "C", "D"]


def parse_answer(text):
    if text is None:
        return None

    t = text.strip()
    patterns = [
        r"Answer\s*[:：]\s*([ABCD])",
        r"答案\s*[:：]\s*([ABCD])",
        r"选项\s*([ABCD])",
        r"\b([ABCD])\b",
    ]

    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    return None


def is_rag_vqa_sample(sample):
    return sample.get("task_type") == RAG_TASK_TYPE


def sorted_rag_images(sample):
    return sorted(sample["images"], key=lambda image: int(image["position"]))


def sorted_text_options(sample):
    by_label = {option["label"]: option for option in sample["text_options"]}
    return [by_label[label] for label in ANSWER_LABELS if label in by_label]


def build_messages(sample, project_root):
    if is_rag_vqa_sample(sample):
        return build_rag_vqa_messages(sample, project_root)
    return build_image_choice_messages(sample, project_root)


def build_image_choice_messages(sample, project_root):
    content = []

    for opt in sample["options"]:
        img_path = str((Path(project_root) / opt["path"]).resolve())
        content.append({"type": "text", "text": f"Option {opt['label']}:"})
        content.append({"type": "image", "image": img_path})

    prompt = (
        f"Question: {sample['question']}\n"
        "There are four image options: A, B, C, and D.\n"
        "Choose the single image option that best answers the question.\n"
        "You must answer in this exact format:\n"
        "Answer: <A/B/C/D>\n"
        "Evidence: <one short visual reason>\n"
    )
    content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]


def build_rag_vqa_prompt(sample):
    option_lines = [
        f"{option['label']}. {option['text']}" for option in sorted_text_options(sample)
    ]
    return (
        "You are given four images: Image 1, Image 2, Image 3, and Image 4. "
        "Only one of these images is relevant to the question. Use the relevant "
        "visual evidence to answer the question.\n\n"
        f"Question: {sample['question']}\n\n"
        "Text answer options:\n"
        + "\n".join(option_lines)
        + "\n\n"
        "Please choose the correct text answer option. Do not answer with Image 1, "
        "Image 2, Image 3, or Image 4.\n"
        "Your response must follow this format:\n"
        "Answer: <A/B/C/D>\n"
        "Evidence: <one short visual reason>\n"
    )


def build_rag_vqa_messages(sample, project_root):
    content = []

    for image in sorted_rag_images(sample):
        img_path = str((Path(project_root) / image["path"]).resolve())
        label = image.get("label", f"Image {image['position']}")
        content.append({"type": "text", "text": f"{label}:"})
        content.append({"type": "image", "image": img_path})

    content.append({"type": "text", "text": build_rag_vqa_prompt(sample)})
    return [{"role": "user", "content": content}]


def load_jsonl(path, limit=None):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))

    return rows


def mark_clean_sample(sample):
    sample = copy.deepcopy(sample)
    sample["corruption"] = "clean"
    sample["severity"] = 0
    sample["corruption_scope"] = "none"
    return sample


def corrupt_legacy_sample(raw_sample, project_root, args, idx):
    if corrupt_option_images is None:
        if args.corruption == "clean":
            return mark_clean_sample(raw_sample)
        raise RuntimeError(
            "methods.corruption.corrupt_option_images is unavailable; "
            "only --corruption clean can run without it."
        )

    return corrupt_option_images(
        raw_sample,
        project_root=project_root,
        corruption=args.corruption,
        severity=args.severity,
        only_positive=args.only_positive,
        seed=args.seed + idx,
    )


def rag_images_to_corruption_options(sample):
    options = []
    for image in sorted_rag_images(sample):
        option = copy.deepcopy(image)
        option["is_correct"] = bool(image.get("is_relevant", False))
        options.append(option)
    return options


def corruption_options_to_rag_images(options):
    images = []
    for option in options:
        image = copy.deepcopy(option)
        image["is_relevant"] = bool(option.get("is_relevant", option.get("is_correct")))
        image.pop("is_correct", None)
        images.append(image)
    return sorted(images, key=lambda image: int(image["position"]))


def corrupt_rag_sample(raw_sample, project_root, args, idx):
    if corrupt_option_images is None:
        if args.corruption == "clean":
            sample = mark_clean_sample(raw_sample)
            sample["images"] = sorted_rag_images(sample)
            sample["options"] = copy.deepcopy(sample["images"])
            return sample
        raise RuntimeError(
            "methods.corruption.corrupt_option_images is unavailable; "
            "only --corruption clean can run without it."
        )

    corruption_sample = copy.deepcopy(raw_sample)
    corruption_sample["options"] = rag_images_to_corruption_options(raw_sample)
    corruption_sample["answer_image_id"] = raw_sample["relevant_image_id"]

    corrupted = corrupt_option_images(
        corruption_sample,
        project_root=project_root,
        corruption=args.corruption,
        severity=args.severity,
        only_positive=args.only_positive,
        seed=args.seed + idx,
    )
    corrupted["images"] = corruption_options_to_rag_images(corrupted["options"])
    corrupted["options"] = copy.deepcopy(corrupted["images"])
    corrupted.pop("answer_image_id", None)
    return corrupted


def prepare_sample(raw_sample, project_root, args, idx):
    if is_rag_vqa_sample(raw_sample):
        return corrupt_rag_sample(raw_sample, project_root, args, idx)
    return corrupt_legacy_sample(raw_sample, project_root, args, idx)


def output_record(sample, prediction, raw_response):
    record = copy.deepcopy(sample)
    record["prediction"] = prediction
    record["is_correct"] = prediction == sample["answer"]
    record["raw_response"] = raw_response
    record["corruption"] = sample.get("corruption", "clean")
    record["severity"] = sample.get("severity", 0)
    record["corruption_scope"] = sample.get("corruption_scope", "none")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--corruption",
        type=str,
        default="clean",
        choices=["clean", "blur", "brightness", "noise", "occlusion"],
    )
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--only_positive", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model:", args.model_path)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    rows = load_jsonl(args.input_jsonl, limit=args.limit)

    with out_file.open("w", encoding="utf-8") as fout:
        for idx, raw_sample in enumerate(tqdm(rows, desc="evaluating")):
            sample = prepare_sample(raw_sample, project_root, args, idx)
            messages = build_messages(sample, project_root)

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            pred = parse_answer(response)
            record = output_record(sample, pred, response)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print("Wrote:", out_file)


if __name__ == "__main__":
    main()
