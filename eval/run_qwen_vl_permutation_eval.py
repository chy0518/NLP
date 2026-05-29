import argparse
import itertools
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from methods.corruption import corrupt_option_images

OPTION_LABELS = ("A", "B", "C", "D")


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

    for pattern in patterns:
        match = re.search(pattern, t, flags=re.IGNORECASE)
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
        base_id = row["base_id"]
        if base_id in seen:
            continue
        seen.add(base_id)
        bases.append(row)
        if limit_bases is not None and len(bases) >= limit_bases:
            break

    return bases


def relabel_options(options):
    relabeled = []
    for label, option in zip(OPTION_LABELS, options):
        new_option = dict(option)
        new_option["label"] = label
        relabeled.append(new_option)
    return relabeled


def make_permuted_sample(base_sample, permuted_options, perm_index, corruption, severity):
    answer_image_id = int(base_sample["answer_image_id"])
    answer = None

    for option in permuted_options:
        if int(option["image_id"]) == answer_image_id:
            answer = option["label"]
            option["is_correct"] = True
        else:
            option["is_correct"] = False

    if answer is None:
        raise ValueError(f"answer image not found for {base_sample['base_id']}")

    sample = dict(base_sample)
    sample["sample_id"] = f"{base_sample['base_id']}_perm_{perm_index:03d}"
    sample["options"] = permuted_options
    sample["answer"] = answer
    sample["positive_position"] = answer
    sample["permutation_index"] = perm_index
    sample["permutation_image_ids"] = [int(opt["image_id"]) for opt in permuted_options]
    sample["corruption"] = corruption
    sample["severity"] = 0 if corruption == "clean" else severity
    return sample


def generate_permuted_samples(base_sample, num_permutations, seed, corruption, severity):
    options = [dict(option) for option in base_sample["options"]]
    all_perms = list(itertools.permutations(options))
    rng = random.Random(seed)

    if num_permutations > len(all_perms):
        raise ValueError(
            f"num_permutations must be <= {len(all_perms)} for four options"
        )

    if num_permutations == len(all_perms):
        selected = all_perms
    else:
        selected = rng.sample(all_perms, num_permutations)

    samples = []
    for perm_index, perm in enumerate(selected):
        permuted_options = relabel_options(perm)
        samples.append(
            make_permuted_sample(
                base_sample,
                permuted_options,
                perm_index,
                corruption=corruption,
                severity=severity,
            )
        )
    return samples


def build_messages(sample, project_root):
    content = []

    for option in sample["options"]:
        img_path = str((Path(project_root) / option["path"]).resolve())
        content.append({"type": "text", "text": f"Option {option['label']}:"})
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL on K full image permutations per base sample."
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
            desc="permutation evaluating",
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
                sample = corrupt_option_images(
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

                pred = parse_answer(response)
                is_correct = pred == sample["answer"]

                record = {
                    "sample_id": sample["sample_id"],
                    "base_id": sample["base_id"],
                    "split": sample.get("split"),
                    "task_type": sample.get("task_type"),
                    "target_category": sample.get("target_category"),
                    "caption": sample.get("caption"),
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "answer_image_id": sample["answer_image_id"],
                    "positive_position": sample["positive_position"],
                    "permutation_index": sample["permutation_index"],
                    "permutation_image_ids": sample["permutation_image_ids"],
                    "corruption": sample.get("corruption", "clean"),
                    "severity": sample.get("severity", 0),
                    "corruption_scope": sample.get("corruption_scope", "none"),
                    "prediction": pred,
                    "is_correct": is_correct,
                    "raw_response": response,
                    "evidence": extract_evidence(response),
                    "options": sample["options"],
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                progress.update(1)

        progress.close()

    print("Wrote:", out_file)


if __name__ == "__main__":
    main()
