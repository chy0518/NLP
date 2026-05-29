# OrderGuard++ COCO Data

This directory builds a simplified OrderGuard++ development and test set from
COCO `val2017`.

Each base sample has one target category, one positive image containing that
category, and three negative images that do not contain that category. Each base
sample is expanded into four positioned versions where the positive image
appears at `A`, `B`, `C`, and `D`.

## Download COCO

Run from the `OrderGuard` directory:

```bash
bash scripts/download_coco_val2017.sh
```

The expected files are:

```text
data/coco/val2017/
data/coco/annotations/instances_val2017.json
data/coco/annotations/captions_val2017.json
```

## Build Object VQA Splits

Run:

```bash
python3 data/coco_orderguard_builder.py
```

Default output:

```text
data/orderguard_dev.jsonl
data/orderguard_test.jsonl
```

Default scale:

```text
dev: 100 base samples -> 400 JSONL rows
test: 300 base samples -> 1200 JSONL rows
```

The builder uses a fixed random seed by default and writes relative image paths
such as `data/coco/val2017/000000000123.jpg`.

## Check Output

Run:

```bash
wc -l data/orderguard_dev.jsonl data/orderguard_test.jsonl
```

Expected:

```text
400 data/orderguard_dev.jsonl
1200 data/orderguard_test.jsonl
1600 total
```

You can inspect one row with:

```bash
head -n 1 data/orderguard_dev.jsonl | python3 -m json.tool
```

For command options:

```bash
python3 data/coco_orderguard_builder.py --help
```

## Build Caption Matching Hard Splits

Run:

```bash
python3 data/coco_caption_orderguard_builder.py
```

Default output:

```text
data/orderguard_caption_hard_dev.jsonl
data/orderguard_caption_hard_test.jsonl
```

Default scale:

```text
hard_dev: 30 base samples -> 120 JSONL rows
hard_test: 100 base samples -> 400 JSONL rows
```

Each base sample uses one COCO caption and four image options. The positive
image is the original image for that caption. The three negative images are
randomly sampled COCO `val2017` images and never include the positive image.
Each base sample is expanded so that the positive image appears at `A`, `B`,
`C`, and `D`.

The JSONL schema follows the object VQA files and adds:

```text
task_type: "caption_matching"
caption: original COCO caption
```

For caption matching rows, `target_category` is `null` because the task is based
on a full caption rather than an object category.

Check output:

```bash
wc -l data/orderguard_caption_hard_dev.jsonl data/orderguard_caption_hard_test.jsonl
head -n 1 data/orderguard_caption_hard_dev.jsonl | python3 -m json.tool
```

Expected:

```text
120 data/orderguard_caption_hard_dev.jsonl
400 data/orderguard_caption_hard_test.jsonl
520 total
```

For command options:

```bash
python3 data/coco_caption_orderguard_builder.py --help
```

## Build Caption Matching Hard-Negative Splits

The random caption matching split above can be too easy for strong VLMs because
the three negative images are random COCO images. The hard-negative split uses
`instances_val2017.json` to choose visually similar negative images that share
COCO object categories with the positive image.

Run:

```bash
python data/coco_caption_hard_negative_builder.py \
  --coco_root data/coco \
  --caption_file data/coco/annotations/captions_val2017.json \
  --instance_file data/coco/annotations/instances_val2017.json \
  --out_dir data \
  --num_dev 30 \
  --num_test 100 \
  --seed 20260528 \
  --top_k_hard_negatives 50
```

Default output:

```text
data/orderguard_caption_hardneg_dev.jsonl
data/orderguard_caption_hardneg_test.jsonl
```

Default scale:

```text
caption_hardneg_dev: 30 base samples -> 120 JSONL rows
caption_hardneg_test: 100 base samples -> 400 JSONL rows
```

Each base sample uses one COCO caption and the caption's original image as the
positive image. The builder filters to captions with at least 7 words and
positive images with at least 2 COCO object categories. Negative images must not
be the positive image and are selected from images sharing at least one object
category with the positive image. Candidates are ranked by shared category count
and Jaccard category similarity, then sampled from the top-K hardest candidates.

The JSONL schema follows the other OrderGuard files and adds:

```text
task_type: "caption_matching_hard_negative"
caption: original COCO caption
positive_categories: positive image COCO object categories
options[].hard_negative_score: Jaccard category similarity for negative options
options[].shared_categories: category names shared with the positive image
```

Check output:

```bash
wc -l data/orderguard_caption_hardneg_dev.jsonl data/orderguard_caption_hardneg_test.jsonl
head -n 1 data/orderguard_caption_hardneg_dev.jsonl | python3 -m json.tool
```

Expected:

```text
120 data/orderguard_caption_hardneg_dev.jsonl
400 data/orderguard_caption_hardneg_test.jsonl
520 total
```

For command options:

```bash
python3 data/coco_caption_hard_negative_builder.py --help
```

## Build Semantic-Visual Hard Negative Caption Matching Set

This split is harder than the category-only hard-negative set. It ranks negative
images with a joint score:

```text
hard_score = alpha * caption_similarity
           + beta * image_similarity
           + gamma * category_overlap_score
```

Defaults:

```text
alpha = 0.45
beta = 0.35
gamma = 0.20
caption_model_name = sentence-transformers/all-MiniLM-L6-v2
clip_model_name = openai/clip-vit-base-patch32
```

The builder caches embeddings under:

```text
data/cache/caption_embeddings_semvis.pkl
data/cache/image_embeddings_semvis.pkl
```

Install dependencies before running this builder:

```bash
pip install sentence-transformers transformers pillow tqdm numpy scikit-learn
```

If you want to use an OpenCLIP-based workflow separately:

```bash
pip install open_clip_torch
```

Full semantic-visual command:

```bash
python data/coco_caption_semantic_visual_hard_builder.py \
  --coco_root data/coco \
  --caption_file data/coco/annotations/captions_val2017.json \
  --instance_file data/coco/annotations/instances_val2017.json \
  --out_dir data \
  --num_dev 30 \
  --num_test 100 \
  --seed 20260528 \
  --top_k_hard_negatives 50 \
  --alpha 0.45 \
  --beta 0.35 \
  --gamma 0.20
```

If CLIP image embedding dependencies or model downloads are not available, use
the degraded semantic-category version:

```bash
python data/coco_caption_semantic_visual_hard_builder.py \
  --coco_root data/coco \
  --caption_file data/coco/annotations/captions_val2017.json \
  --instance_file data/coco/annotations/instances_val2017.json \
  --out_dir data \
  --num_dev 30 \
  --num_test 100 \
  --seed 20260528 \
  --top_k_hard_negatives 50 \
  --disable_image_similarity
```

The degraded version still uses sentence-transformer caption similarity and COCO
category overlap, and writes `image_similarity: 0.0` for negative options.

Default output:

```text
data/orderguard_caption_semvis_hard_dev.jsonl
data/orderguard_caption_semvis_hard_test.jsonl
```

Default scale:

```text
caption_semvis_hard_dev: 30 base samples -> 120 JSONL rows
caption_semvis_hard_test: 100 base samples -> 400 JSONL rows
```

Check output:

```bash
wc -l data/orderguard_caption_semvis_hard_dev.jsonl data/orderguard_caption_semvis_hard_test.jsonl
head -n 1 data/orderguard_caption_semvis_hard_test.jsonl | python3 -m json.tool
```

Expected:

```text
120 data/orderguard_caption_semvis_hard_dev.jsonl
400 data/orderguard_caption_semvis_hard_test.jsonl
520 total
```

For command options:

```bash
python3 data/coco_caption_semantic_visual_hard_builder.py --help
```
