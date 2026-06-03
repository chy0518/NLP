# OrderGuard: Order Sensitivity in Multi-image VQA

本仓库是自然语言处理课程项目代码，目标是在论文 **Order Matters** 的启发下，复现并改进多模态大模型在多图输入场景中的顺序敏感性实验。

当前实验主线已经从早期的“四图选择 caption matching”扩展到更接近论文 RAG-VQA 设置的 **RAG-style Multi-image VQA**：输入 4 张图、一个问题和 4 个文本答案选项，模型需要基于其中一张 question-relevant image 回答文本选项 `A/B/C/D`。我们发现 Qwen2.5-VL 在该任务上存在明显的 relevant-image position sensitivity，并进一步实现了基于位置可靠性的 permutation answer voting。

## 当前重点

### 1. Balanced 4-position Semantic+Category HardNeg

用于排除原始四图选择任务中 negative image 顺序不均衡带来的 confound。每个 base sample 会展开成 4 个位置版本，并保证每张图在同一个 base sample 内都恰好出现在 `A/B/C/D` 各一次。

主要文件：

- `data/orderguard_caption_semvis_hard_large_balanced_dev.jsonl`
- `data/orderguard_caption_semvis_hard_large_balanced_test.jsonl`
- `data_builders/coco_caption_balanced_position_builder.py`

### 2. RAG-style Multi-image VQA

这是当前报告和方法改进的重点。每个样本包含：

- 4 张图片；
- 1 张 question-relevant image；
- 3 张 hard distractor images；
- 1 个关于 relevant image 的问题；
- 4 个文本答案选项 `A/B/C/D`。

注意：这里的 `Image 1/2/3/4` 是图片位置，`A/B/C/D` 是文本答案选项，二者不是同一件事。

主要文件：

- `data/orderguard_rag_vqa_dev.jsonl`
- `data/orderguard_rag_vqa_test.jsonl`
- `data_builders/coco_rag_style_vqa_builder.py`

## 方法

### Caption/Image-choice 任务

已实现：

- `single_order_mean`
- `permutation_voting`
- `position_calibrated`
- `evidence_orderguard`

对应入口：

- `eval/run_orderguard_methods.py`
- `methods/aggregation.py`
- `methods/position_calibration.py`
- `methods/evidence_orderguard.py`

### RAG-style VQA 任务

已实现：

- 普通 permutation answer voting；
- position weighted answer voting；
- `raw` / `power` / `threshold` / `softmax` / `logit` 等权重变换；
- 使用 dev permutation 结果估计不同 relevant-image position 的可靠性，再在 test 上聚合答案。

对应入口：

- `eval/run_qwen_vl_rag_permutation_eval.py`
- `eval/run_rag_answer_voting.py`

当前效果最好的设置是 softmax-weighted voting，核心思想是：如果 dev set 上 relevant image 位于某个位置时模型更可靠，那么 test 聚合时该位置产生的回答应获得更高权重。

## 最新关键结果

| Dataset / Setting | Accuracy | All-4 Consistency | Unstable Bases | Full-set Gap | Unstable-only Gap |
|---|---:|---:|---:|---:|---:|
| Balanced Semantic+Category Clean | 87.58% | 83.00% | 51 / 300 | 4.67% | 27.45% |
| Balanced Semantic+Category Occlusion | 85.33% | 79.33% | 62 / 300 | 4.33% | 20.97% |
| RAG-style VQA Clean | 48.25% | 32.50% | 135 / 200 | 35.00% | 51.85% |

RAG-style VQA 中的 relevant-image position accuracy：

| Relevant image position | Accuracy |
|---|---:|
| Image 1 | 73.00% |
| Image 2 | 41.50% |
| Image 3 | 38.00% |
| Image 4 | 40.50% |

RAG-style permutation voting 当前结果：

| Method | Accuracy |
|---|---:|
| Original row-level Qwen outputs | 48.5% |
| Majority permutation voting | 43.5% |
| Raw position weighted voting | 51.5% |
| Power weighted voting, alpha=3 | 69.5% |
| Threshold voting, threshold=0.6 | 75.0% |
| Softmax voting, temperature=0.1 | 74.5% |

## 服务器运行示例

服务器中项目通常位于：

```bash
cd /root/autodl-tmp/OrderGuard/NLP
```

Qwen2.5-VL 模型路径示例：

```bash
MODEL=/root/models/Qwen2.5-VL-7B-Instruct
```

### 1. 运行 RAG-VQA permutation 推理

```bash
PYTHONPATH=. python eval/run_qwen_vl_rag_permutation_eval.py \
  --model_path $MODEL \
  --input_jsonl data/orderguard_rag_vqa_test.jsonl \
  --output_jsonl result/rag_vqa_test_clean_perm12_qwen25vl.jsonl \
  --project_root . \
  --corruption clean \
  --num_permutations 12 \
  --max_new_tokens 64
```

dev set 也需要跑一遍，用于估计位置可靠性：

```bash
PYTHONPATH=. python eval/run_qwen_vl_rag_permutation_eval.py \
  --model_path $MODEL \
  --input_jsonl data/orderguard_rag_vqa_dev.jsonl \
  --output_jsonl result/rag_vqa_dev_clean_perm12_qwen25vl.jsonl \
  --project_root . \
  --corruption clean \
  --num_permutations 12 \
  --max_new_tokens 64
```

### 2. 运行 RAG-VQA answer voting

普通 majority voting：

```bash
PYTHONPATH=. python eval/run_rag_answer_voting.py \
  --input_jsonl result/rag_vqa_test_clean_perm12_qwen25vl.jsonl \
  --output_jsonl result/rag_vqa_test_clean_perm12_answer_voting.jsonl \
  --metrics_json result/rag_vqa_test_clean_perm12_answer_voting_metrics.json
```

dev 权重 + softmax voting：

```bash
PYTHONPATH=. python eval/run_rag_answer_voting.py \
  --input_jsonl result/rag_vqa_test_clean_perm12_qwen25vl.jsonl \
  --weight_jsonl result/rag_vqa_dev_clean_perm12_qwen25vl.jsonl \
  --weight_mode softmax \
  --temperature 0.1 \
  --output_jsonl result/rag_vqa_test_clean_perm12_vote_softmax01.jsonl \
  --metrics_json result/rag_vqa_test_clean_perm12_vote_softmax01_metrics.json
```

### 3. 启动可视化 demo

```bash
streamlit run demo/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

demo 默认读取：

- `result/rag_vqa_test_clean_perm12_qwen25vl.jsonl`
- `result/rag_vqa_test_clean_perm12_vote_softmax01.jsonl`

如果文件名不同，可以在 Streamlit 左侧栏手动修改。demo 适合展示同一个 base sample 在不同 image permutation 下的模型回答，并寻找 “majority wrong, weighted correct” 的代表案例。

如果服务器出现 `starlette.middleware.gzip` 相关报错，可尝试：

```bash
pip install -U "streamlit>=1.45" "starlette>=0.46"
```

## 项目结构

```text
NLP/
├── data/                         # 构造好的 JSONL 数据集
├── data_builders/                # COCO 数据集构建脚本
├── demo/
│   └── app.py                    # Streamlit 可视化 demo
├── docs/
│   └── ORDERGUARD_METHODS.md     # 方法与运行命令说明
├── eval/
│   ├── run_qwen_vl_eval.py
│   ├── run_qwen_vl_rag_permutation_eval.py
│   ├── run_rag_answer_voting.py
│   └── run_orderguard_methods.py
├── methods/                      # 聚合与校准方法
├── models/                       # 模型调用封装
├── result/                       # 实验输出
└── tests/                        # 单元测试
```

## 更多说明

- 最新实验交接说明：`README_experiment_handoff.md`
- 方法与命令细节：`docs/ORDERGUARD_METHODS.md`
- 数据格式说明：`docs/DATASET_FORMAT.md`

