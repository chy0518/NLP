# OrderGuard++ 数据集格式说明

这个文件用于说明 `data/orderguard_dev.jsonl` 和
`data/orderguard_test.jsonl` 的数据格式，方便同学直接读取、评测，或把
数据集说明复制给 AI 继续交流。

## 数据集目标

OrderGuard++ 是一个基于 COCO `val2017` 构造的 4-image object VQA
multiple-choice 数据集，用来测试多模态模型是否会受到图片顺序影响。

每个问题有 4 个图片选项 `A/B/C/D`，问题形式是：

```text
Which image contains a {target_category}?
```

其中只有 1 张图包含目标类别，其余 3 张图都不包含目标类别。模型需要回答
正确图片所在的选项标签，例如 `A`、`B`、`C` 或 `D`。

## 文件

```text
data/orderguard_dev.jsonl
data/orderguard_test.jsonl
data/orderguard_caption_hard_dev.jsonl
data/orderguard_caption_hard_test.jsonl
data/orderguard_caption_hardneg_dev.jsonl
data/orderguard_caption_hardneg_test.jsonl
data/orderguard_caption_semvis_hard_dev.jsonl
data/orderguard_caption_semvis_hard_test.jsonl
```

默认规模：

```text
dev: 100 个 base samples，展开后 400 行
test: 300 个 base samples，展开后 1200 行
caption hard_dev: 30 个 base samples，展开后 120 行
caption hard_test: 100 个 base samples，展开后 400 行
caption hardneg_dev: 30 个 base samples，展开后 120 行
caption hardneg_test: 100 个 base samples，展开后 400 行
caption semvis hard_dev: 30 个 base samples，展开后 120 行
caption semvis hard_test: 100 个 base samples，展开后 400 行
```

每一行是一个完整的 JSON 对象。JSONL 的意思是 JSON Lines，也就是一行一个
JSON，不是一个大 JSON 数组。

## 样本构造逻辑

每个 base sample 包含：

- 1 个 `target_category`
- 1 张 positive image：包含 `target_category`
- 3 张 negative images：不包含 `target_category`
- 1 个问题：`Which image contains a {target_category}?`

然后每个 base sample 会展开成 4 行：

```text
正确图在 A
正确图在 B
正确图在 C
正确图在 D
```

因此，同一个 `base_id` 会对应 4 个 `sample_id`。这 4 行使用同一张正确图和
同一组干扰图，只是选项顺序不同。

## 目标类别

当前使用 16 个 COCO 类别：

```text
dog, cat, car, bicycle, person, bus, train, horse, sheep, bird, boat,
chair, couch, dining table, laptop, cell phone
```

## 字段说明

一行样本大致如下：

```json
{
  "sample_id": "dev_000001_pos_A",
  "base_id": "dev_base_000001",
  "split": "dev",
  "target_category": "cell phone",
  "question": "Which image contains a cell phone?",
  "options": [
    {
      "label": "A",
      "image_id": 553731,
      "file_name": "000000553731.jpg",
      "path": "data/coco/val2017/000000553731.jpg",
      "is_correct": true
    }
  ],
  "answer": "A",
  "answer_image_id": 553731,
  "positive_position": "A"
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 当前 positioned sample 的唯一 ID，例如正确图在 A 的版本。 |
| `base_id` | base sample 的 ID。同一个 base sample 会展开成 4 行。 |
| `split` | 数据划分，取值为 `dev` 或 `test`。 |
| `target_category` | 要寻找的 COCO 目标类别。 |
| `question` | 给模型的问题。 |
| `options` | 4 个图片选项，顺序即模型看到的 A/B/C/D 顺序。 |
| `options[].label` | 选项标签，取值为 `A`、`B`、`C`、`D`。 |
| `options[].image_id` | COCO image id。 |
| `options[].file_name` | COCO 图片文件名。 |
| `options[].path` | 相对路径，从 `OrderGuard/` 目录出发。 |
| `options[].is_correct` | 该选项是否为正确图片。评测时通常不要提供给模型。 |
| `answer` | 正确选项标签。评测时作为 gold answer。 |
| `answer_image_id` | 正确图片的 COCO image id。 |
| `positive_position` | 正确图所在位置，与 `answer` 一致。 |

## 路径说明

JSONL 中的图片路径是相对路径，例如：

```text
data/coco/val2017/000000553731.jpg
```

如果你的代码从 `OrderGuard/` 目录运行，可以直接使用这个路径。

如果你的代码从别的目录运行，需要拼接 `OrderGuard` 的绝对路径，例如：

```python
from pathlib import Path

orderguard_root = Path("/path/to/OrderGuard")
image_path = orderguard_root / row["options"][0]["path"]
```

## 评测时不要泄露的字段

给模型输入时，通常只应该提供：

- `question`
- `options[].label`
- `options[].path` 或实际图片内容

不要把下面字段给模型：

- `options[].is_correct`
- `answer`
- `answer_image_id`
- `positive_position`

这些字段是评测用的 gold label。

## Caption Matching Hard Set

caption matching hard set 的文件是：

```text
data/orderguard_caption_hard_dev.jsonl
data/orderguard_caption_hard_test.jsonl
```

它和 object VQA JSONL 使用相同的主字段，并额外加入：

| 字段 | 含义 |
| --- | --- |
| `task_type` | 固定为 `caption_matching`。 |
| `caption` | 原始 COCO caption。 |

这个任务不使用 object category，所以 `target_category` 为 `null`。

caption matching 的问题格式是：

```text
Which image best matches the caption: "{caption}"?
```

每个 base sample 中，positive image 是这条 caption 对应的 COCO 原图，3 张
negative images 是其他随机 COCO 图片，并且不包含 positive image。和 object VQA
一样，每个 base sample 会展开成 4 行，让正确图分别出现在 `A/B/C/D`。

评测 caption matching 时，给模型输入 `question` 和 4 张图片即可，不要泄露：

- `options[].is_correct`
- `answer`
- `answer_image_id`
- `positive_position`

## Caption Matching Hard Negative Set

hard negative caption matching 的文件是：

```text
data/orderguard_caption_hardneg_dev.jsonl
data/orderguard_caption_hardneg_test.jsonl
```

它和随机 negative caption matching 的区别是：随机版本的 3 张 negative images
是随机 COCO 图片，强模型可能很容易排除；hard negative 版本会用
`instances_val2017.json` 的 object categories 选择视觉类别相似的图片作为干扰项。

构造规则：

- positive image 是 caption 对应的 COCO 原图。
- caption 至少 7 个单词。
- positive image 至少包含 2 个 COCO object categories。
- negative image 不能是 positive image 自己。
- negative image 需要和 positive image 共享至少 1 个 object category。
- 候选 negative images 会按共享类别数和 Jaccard category similarity 排序。
- 最终从 top-K hardest candidates 中随机采样 3 张，默认 `top_k_hard_negatives=50`。

每行在通用字段基础上额外包含：

| 字段 | 含义 |
| --- | --- |
| `task_type` | 固定为 `caption_matching_hard_negative`。 |
| `caption` | 原始 COCO caption。 |
| `positive_categories` | positive image 的 COCO object categories。 |
| `options[].hard_negative_score` | 仅 negative option 有，表示与 positive image 的 category Jaccard similarity。 |
| `options[].shared_categories` | 仅 negative option 有，表示该 negative 与 positive image 共享的类别。 |

示例结构：

```json
{
  "sample_id": "caption_hardneg_test_000001_pos_A",
  "base_id": "caption_hardneg_test_base_000001",
  "split": "caption_hardneg_test",
  "task_type": "caption_matching_hard_negative",
  "caption": "A man riding a bicycle down a city street.",
  "target_category": null,
  "question": "Which image best matches the caption: \"A man riding a bicycle down a city street.\"?",
  "options": [
    {
      "label": "A",
      "image_id": 123,
      "file_name": "000000000123.jpg",
      "path": "data/coco/val2017/000000000123.jpg",
      "is_correct": true
    },
    {
      "label": "B",
      "image_id": 456,
      "file_name": "000000000456.jpg",
      "path": "data/coco/val2017/000000000456.jpg",
      "is_correct": false,
      "hard_negative_score": 0.5,
      "shared_categories": ["person", "bicycle"]
    }
  ],
  "answer": "A",
  "answer_image_id": 123,
  "positive_position": "A",
  "positive_categories": ["person", "bicycle"]
}
```

评测时仍然只把 `question` 和 4 张图片给模型。`hard_negative_score`、
`shared_categories` 和 `positive_categories` 是分析字段，不建议作为模型输入。

## Semantic-Visual Hard Negative Caption Matching Set

semantic-visual hard negative caption matching 的文件是：

```text
data/orderguard_caption_semvis_hard_dev.jsonl
data/orderguard_caption_semvis_hard_test.jsonl
```

这个版本比 `caption_hardneg` 更难。`caption_hardneg` 只看 COCO object category
overlap；`caption_semvis_hard` 会联合使用：

- `caption_similarity`：positive caption 和 candidate image representative caption 的 sentence-transformer cosine similarity。
- `image_similarity`：positive image 和 candidate image 的 CLIP image embedding cosine similarity。
- `category_overlap_score`：positive image categories 和 candidate image categories 的 Jaccard similarity。

最终分数：

```text
hard_score = 0.45 * caption_similarity
           + 0.35 * image_similarity
           + 0.20 * category_overlap_score
```

如果运行时使用 `--disable_image_similarity`，则不加载 CLIP，`image_similarity`
会写为 `0.0`。这个降级版本仍然比只看 category overlap 更难，因为它会使用
caption semantic similarity。

每行在通用字段基础上额外包含：

| 字段 | 含义 |
| --- | --- |
| `task_type` | 固定为 `caption_matching_semantic_visual_hard_negative`。 |
| `caption` | positive image 对应的 COCO caption。 |
| `positive_categories` | positive image 的 COCO object categories。 |
| `options[].hard_negative_score` | 仅 negative option 有，最终 hard negative 分数。 |
| `options[].caption_similarity` | 仅 negative option 有，caption semantic similarity。 |
| `options[].image_similarity` | 仅 negative option 有，CLIP image similarity；禁用图像相似度时为 `0.0`。 |
| `options[].category_overlap_score` | 仅 negative option 有，COCO category Jaccard overlap。 |
| `options[].shared_categories` | 仅 negative option 有，该 negative 与 positive image 共享的类别。 |

示例结构：

```json
{
  "sample_id": "caption_semvis_hard_test_000001_pos_A",
  "base_id": "caption_semvis_hard_test_base_000001",
  "split": "caption_semvis_hard_test",
  "task_type": "caption_matching_semantic_visual_hard_negative",
  "caption": "A man riding a bicycle down a city street.",
  "target_category": null,
  "question": "Which image best matches the caption: \"A man riding a bicycle down a city street.\"?",
  "options": [
    {
      "label": "A",
      "image_id": 123,
      "file_name": "000000000123.jpg",
      "path": "data/coco/val2017/000000000123.jpg",
      "is_correct": true
    },
    {
      "label": "B",
      "image_id": 456,
      "file_name": "000000000456.jpg",
      "path": "data/coco/val2017/000000000456.jpg",
      "is_correct": false,
      "hard_negative_score": 0.7345,
      "caption_similarity": 0.71,
      "image_similarity": 0.68,
      "category_overlap_score": 0.5,
      "shared_categories": ["person", "bicycle"]
    }
  ],
  "answer": "A",
  "answer_image_id": 123,
  "positive_position": "A",
  "positive_categories": ["person", "bicycle"]
}
```

评测时仍然只把 `question` 和 4 张图片给模型。所有 score、shared categories、
positive categories 都是分析字段，不建议作为模型输入。

## 推荐模型输入格式

给多模态模型时，可以组织成：

```text
Question: Which image contains a cell phone?

Options:
A: data/coco/val2017/000000553731.jpg
B: data/coco/val2017/000000300155.jpg
C: data/coco/val2017/000000369323.jpg
D: data/coco/val2017/000000206838.jpg

Please answer with only one letter: A, B, C, or D.
```

如果模型 API 支持多图输入，应该按 `options` 数组顺序传入图片，并在文本里标明
每张图片对应的标签。

## 顺序敏感性分析

这个数据集的关键不是只看 overall accuracy，还要看同一个 `base_id` 的 4 个
positioned versions 是否稳定。

可以统计：

- 每个位置的 accuracy：`A/B/C/D` 分别的准确率
- 同一个 `base_id` 的 4 个版本是否都答对
- 模型是否偏向回答某个固定位置
- 正确图从 `A` 移到 `D` 后，模型预测是否变化

一个模型如果真的理解图片内容，理论上不应该因为正确图片位置变化而大幅改变
表现。

## 读取示例

```python
import json
from pathlib import Path

path = Path("data/orderguard_dev.jsonl")

with path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        question = row["question"]
        options = [(opt["label"], opt["path"]) for opt in row["options"]]
        answer = row["answer"]
        print(question, options, answer)
        break
```

## 可以直接复制给 AI 的上下文

```text
我有一个 OrderGuard++ COCO JSONL 数据集。每行是一个 4-image multiple-choice
object VQA 样本。字段包括 sample_id, base_id, split, target_category,
question, options, answer, answer_image_id, positive_position。

options 是长度为 4 的数组，每个元素有 label, image_id, file_name, path,
is_correct。label 是 A/B/C/D；path 是相对于 OrderGuard/ 目录的图片路径。

任务是让多模态模型回答 question: "Which image contains a {target_category}?"
模型只能输出 A/B/C/D。评测时 answer 是 gold label。不要把 is_correct,
answer, answer_image_id, positive_position 提供给模型。

同一个 base_id 有 4 行 positioned versions，正确图分别出现在 A/B/C/D，用来分析
模型是否对图片顺序敏感。
```

## 常用检查命令

从 `OrderGuard/` 目录运行：

```bash
wc -l data/orderguard_dev.jsonl data/orderguard_test.jsonl
head -n 1 data/orderguard_dev.jsonl | python3 -m json.tool
```

期望行数：

```text
400 data/orderguard_dev.jsonl
1200 data/orderguard_test.jsonl
1600 total
```
