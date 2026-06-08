import argparse
import copy
import itertools
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

RAG_TASK_TYPES = {
    "rag_style_multi_image_vqa",
    "rag_style_multi_image_vqa_retrieval",
    "rag_style_multi_image_vqa_retrieval_category",
}
ANSWER_LABELS = ("A", "B", "C", "D")
IMAGE_POSITIONS = (1, 2, 3, 4)


def parse_answer(text):
    if text is None:
        return None

    patterns = [
        r"Answer\s*[:：]\s*([ABCD])",
        r"答案\s*[:：]\s*([ABCD])",
        r"选项\s*([ABCD])",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def extract_evidence(text):
    if not text:
        return ""

    match = re.search(
        r"Evidence\s*[:：]\s*(.+?)(?:\n\s*(?:Confidence|Answer)\s*[:：]|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return " ".join(match.group(1).strip().split())
    return ""


def load_jsonl(path, limit_rows=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit_rows is not None and idx >= limit_rows:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unique_base_samples(rows, limit_bases=None):
    seen = set()
    bases = []
    for row in rows:
        if row.get("task_type") not in RAG_TASK_TYPES:
            raise ValueError(f"expected one of {RAG_TASK_TYPES}, got {row.get('task_type')}")

        base_id = row["base_id"]
        if base_id in seen:
            continue
        seen.add(base_id)
        bases.append(row)
        if limit_bases is not None and len(bases) >= limit_bases:
            break
    return bases


def sorted_images(sample):
    return sorted(sample["images"], key=lambda image: int(image["position"]))


def sorted_text_options(sample):
    by_label = {option["label"]: option for option in sample["text_options"]}
    return [by_label[label] for label in ANSWER_LABELS]


def relabel_images(images):
    relabeled = []
    for position, image in zip(IMAGE_POSITIONS, images):
        updated = copy.deepcopy(image)
        updated["position"] = position
        updated["label"] = f"Image {position}"
        relabeled.append(updated)
    return relabeled


def make_permuted_sample(base_sample, permuted_images, perm_index, corruption, severity):
    relevant_position = None
    for image in permuted_images:
        if int(image["image_id"]) == int(base_sample["relevant_image_id"]):
            image["is_relevant"] = True
            relevant_position = int(image["position"])
        else:
            image["is_relevant"] = False

    if relevant_position is None:
        raise ValueError(f"relevant image not found for {base_sample['base_id']}")

    sample = copy.deepcopy(base_sample)
    sample["sample_id"] = f"{base_sample['base_id']}_perm_{perm_index:03d}"
    sample["images"] = permuted_images
    sample["options"] = copy.deepcopy(permuted_images)
    sample["relevant_image_position"] = relevant_position
    sample["positive_position"] = f"Image {relevant_position}"
    sample["permutation_index"] = perm_index
    sample["permutation_image_ids"] = [int(image["image_id"]) for image in permuted_images]
    sample["corruption"] = corruption
    sample["severity"] = 0 if corruption == "clean" else severity
    return sample


def generate_permuted_samples(base_sample, num_permutations, seed, corruption, severity):
    images = [copy.deepcopy(image) for image in sorted_images(base_sample)]
    all_perms = list(itertools.permutations(images))
    rng = random.Random(seed)

    if num_permutations < 1 or num_permutations > len(all_perms):
        raise ValueError(f"num_permutations must be between 1 and {len(all_perms)}")

    selected = (
        all_perms
        if num_permutations == len(all_perms)
        else rng.sample(all_perms, num_permutations)
    )
    samples = []
    for perm_index, perm in enumerate(selected):
        samples.append(
            make_permuted_sample(
                base_sample,
                relabel_images(perm),
                perm_index,
                corruption=corruption,
                severity=severity,
            )
        )
    return samples


def rag_images_to_corruption_options(sample):
    options = []
    for image in sorted_images(sample):
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


def apply_corruption(sample, project_root, corruption, severity, only_positive, seed):
    if corruption == "clean":
        sample = copy.deepcopy(sample)
        sample["corruption"] = "clean"
        sample["severity"] = 0
        sample["corruption_scope"] = "none"
        return sample

    from methods.corruption import corrupt_option_images

    corruption_sample = copy.deepcopy(sample)
    corruption_sample["options"] = rag_images_to_corruption_options(sample)
    corruption_sample["answer_image_id"] = sample["relevant_image_id"]
    corrupted = corrupt_option_images(
        corruption_sample,
        project_root=project_root,
        corruption=corruption,
        severity=severity,
        only_positive=only_positive,
        seed=seed,
    )
    corrupted["images"] = corruption_options_to_rag_images(corrupted["options"])
    corrupted["options"] = copy.deepcopy(corrupted["images"])
    corrupted.pop("answer_image_id", None)
    return corrupted


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


def build_messages(sample, project_root):
    content = []
    for image in sorted_images(sample):
        img_path = str((Path(project_root) / image["path"]).resolve())
        label = image.get("label", f"Image {image['position']}")
        content.append({"type": "text", "text": f"{label}:"})
        content.append({"type": "image", "image": img_path})

    content.append({"type": "text", "text": build_rag_vqa_prompt(sample)})
    return [{"role": "user", "content": content}]


def run_one_sample(model, processor, sample, project_root, max_new_tokens):
    import torch
    from qwen_vl_utils import process_vision_info

    messages = build_messages(sample, project_root)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def output_record(sample, prediction, raw_response):
    record = copy.deepcopy(sample)
    record["prediction"] = prediction
    record["is_correct"] = prediction == sample["answer"]
    record["raw_response"] = raw_response
    record["evidence"] = extract_evidence(raw_response)
    record["corruption"] = sample.get("corruption", "clean")
    record["severity"] = sample.get("severity", 0)
    record["corruption_scope"] = sample.get("corruption_scope", "none")
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-VL on K image permutations per RAG-style VQA base sample."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--num_permutations", type=int, default=12)
    parser.add_argument("--limit_bases", type=int, default=None)
    parser.add_argument("--limit_rows", type=int, default=None)
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

    if args.num_permutations < 1 or args.num_permutations > 24:
        raise ValueError("--num_permutations must be between 1 and 24")

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.input_jsonl, limit_rows=args.limit_rows)
    base_samples = unique_base_samples(rows, limit_bases=args.limit_bases)

    print("Loading model:", args.model_path)
    print("input:", args.input_jsonl)
    print("base_samples:", len(base_samples))
    print("num_permutations:", args.num_permutations)
    print("expected_model_calls:", len(base_samples) * args.num_permutations)

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    with out_file.open("w", encoding="utf-8") as fout:
        progress = tqdm(
            total=len(base_samples) * args.num_permutations,
            desc="rag permutation evaluating",
        )

        for base_idx, base_sample in enumerate(base_samples):
            permuted_samples = generate_permuted_samples(
                base_sample,
                num_permutations=args.num_permutations,
                seed=args.seed + base_idx,
                corruption=args.corruption,
                severity=args.severity,
            )

            for perm_sample in permuted_samples:
                sample_seed = args.seed + base_idx * 1000 + perm_sample["permutation_index"]
                sample = apply_corruption(
                    perm_sample,
                    project_root=project_root,
                    corruption=args.corruption,
                    severity=args.severity,
                    only_positive=args.only_positive,
                    seed=sample_seed,
                )
                response = run_one_sample(
                    model,
                    processor,
                    sample,
                    project_root=project_root,
                    max_new_tokens=args.max_new_tokens,
                )
                prediction = parse_answer(response)
                record = output_record(sample, prediction, response)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                progress.update(1)

        progress.close()

    print("Wrote:", out_file)


if __name__ == "__main__":
    main()
