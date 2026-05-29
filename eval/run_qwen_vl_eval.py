import argparse
import json
import re
from pathlib import Path
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from methods.corruption import corrupt_option_images


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


def build_messages(sample, project_root):
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


def load_jsonl(path, limit=None):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--corruption", type=str, default="clean",
                        choices=["clean", "blur", "brightness", "noise", "occlusion"])
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
            sample = corrupt_option_images(
                raw_sample,
                project_root=project_root,
                corruption=args.corruption,
                severity=args.severity,
                only_positive=args.only_positive,
                seed=args.seed + idx,
            )

            messages = build_messages(sample, project_root)

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
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
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            pred = parse_answer(response)
            is_correct = (pred == sample["answer"])

            record = {
                "sample_id": sample["sample_id"],
                "base_id": sample["base_id"],
                "split": sample["split"],
                "target_category": sample["target_category"],
                "question": sample["question"],
                "answer": sample["answer"],
                "answer_image_id": sample["answer_image_id"],
                "positive_position": sample["positive_position"],
                "corruption": sample.get("corruption", "clean"),
                "severity": sample.get("severity", 0),
                "corruption_scope": sample.get("corruption_scope", "none"),
                "prediction": pred,
                "is_correct": is_correct,
                "raw_response": response,
                "options": sample["options"],
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print("Wrote:", out_file)


if __name__ == "__main__":
    main()
