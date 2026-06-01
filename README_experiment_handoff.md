# 把上面内容粘到这里
## 新增数据集与最新实验结果

在原有 Semantic + Category HardNeg 数据集基础上，我们进一步补充了两组更严谨 / 更贴近原论文设置的数据集：

1. Balanced 4-position Semantic+Category HardNeg
2. RAG-style Multi-image VQA

这两组数据集分别解决两个问题：

- Balanced 4-position 用来排除原始 4-position 中 negative order 未完全均衡导致的 confound；
- RAG-style Multi-image VQA 用来更贴近原论文 RAG-VQA 的设置，即固定问题和文本答案选项，只移动 question-relevant image 的位置。

---

### 1. Balanced 4-position Semantic+Category HardNeg

#### 数据文件

text data/orderguard_caption_semvis_hard_large_balanced_dev.jsonl    200 rows data/orderguard_caption_semvis_hard_large_balanced_test.jsonl   1200 rows 

对应构建脚本：

text data_builders/coco_caption_balanced_position_builder.py 

#### 构造动机

原始 4-position setting 中，每个 base sample 展开为：

text pos_A: A=positive, B=neg1, C=neg2, D=neg3 pos_B: A=neg1, B=positive, C=neg2, D=neg3 pos_C: A=neg1, B=neg2, C=positive, D=neg3 pos_D: A=neg1, B=neg2, C=neg3, D=positive 

这种构造可以观察模型在不同 positive position 下的表现差异，但存在一个问题：negative images 的位置没有完全均衡。例如，neg1 更常出现在 A，neg3 更常出现在 D。因此，如果模型在原始 4-position 中更容易选择 A，可能混合了三种因素：

1. 真实的位置偏好；
2. 某个 hard negative 本身更有迷惑性；
3. 数据构造导致该 hard negative 经常出现在某个固定位置。

为了排除这种 negative order confound，我们构造了 balanced 4-position 数据集。

#### Balanced layout

对每个 base sample，设：

text P = positive image N1, N2, N3 = three hard negative images 

balanced 版本使用：

text pos_A: A=P,  B=N1, C=N2, D=N3 pos_B: A=N3, B=P,  C=N1, D=N2 pos_C: A=N2, B=N3, C=P,  D=N1 pos_D: A=N1, B=N2, C=N3, D=P 

这样每张图在同一个 base sample 内都会恰好出现在 A/B/C/D 各一次。因此，图像身份和位置被更好地解耦。

#### 构建统计

text Dev base samples: 50 Test base samples: 300 Dev rows: 200 Test rows: 1200 Skipped base samples: 0 All balance checks passed: true 

---

### 2. RAG-style Multi-image VQA

#### 数据文件

text data/orderguard_rag_vqa_dev.jsonl    200 rows data/orderguard_rag_vqa_test.jsonl   800 rows 

对应构建脚本：

text data_builders/coco_rag_style_vqa_builder.py 

#### 构造动机

前面的 caption matching 任务是：

text 给定一个 caption，从四张图中选择最匹配的一张图。 

而原论文中的 RAG-VQA 更接近：

text 给定多张检索图像，其中只有一张图和问题相关； 模型需要基于相关图回答一个文本多选问题。 

因此我们构造了 RAG-style Multi-image VQA 数据集。每个样本包含：

- 4 张图片；
- 其中 1 张是 question-relevant image；
- 3 张是 hard distractor images；
- 1 个关于 relevant image 的问题；
- 4 个文本答案选项 A/B/C/D。

注意：这里图片位置使用 Image 1 / Image 2 / Image 3 / Image 4，文本答案选项仍然使用 A/B/C/D。模型需要回答文本选项，而不是回答图片位置。

#### 样本格式示例

json {   "sample_id": "rag_vqa_test_000001_imgpos_1",   "base_id": "rag_vqa_test_base_000001",   "split": "rag_vqa_test",   "task_type": "rag_style_multi_image_vqa",   "vqa_type": "object_existence",   "question": "Which object is present in the relevant image?",   "answer": "A",   "answer_text": "pizza",   "target_category": "pizza",   "relevant_image_position": 1,   "positive_position": "Image 1",   "text_options": [     {"label": "A", "text": "pizza"},     {"label": "B", "text": "broccoli"},     {"label": "C", "text": "donut"},     {"label": "D", "text": "banana"}   ] } 

#### 构建统计

text Usable base candidates: 11486 Captioned images indexed: 5000 Task specs: 84 Dev base samples: 50 Test base samples: 200 Dev rows: 200 Test rows: 800 Skipped candidate reasons: {'activity_without_person': 83, 'not_enough_text_distractors': 514} Dev answer distribution: {'A': 13, 'B': 13, 'C': 12, 'D': 12} Test answer distribution: {'A': 50, 'B': 50, 'C': 50, 'D': 50} Dev VQA type distribution: {'activity': 2, 'animal_type': 8, 'object_existence': 31, 'vehicle_type': 9} Test VQA type distribution: {'activity': 32, 'animal_type': 14, 'object_existence': 127, 'vehicle_type': 27} Average hard negative score: 0.7089 All balance checks passed: true 

RAG-style VQA 的文本答案分布在 test set 中完全均衡：

text A: 50 B: 50 C: 50 D: 50 

因此，如果模型表现出差异，更主要反映的是 relevant image position 的影响，而不是文本答案 label 分布不均。

---

## 最新实验结果

目前新增数据集上已经完成三组主要实验：

1. Balanced 4-position Clean
2. Balanced 4-position Occlusion
3. RAG-style Multi-image VQA Clean

---

### 1. Balanced 4-position Clean

输出文件：

text result/caption_semvis_hard_large_balanced_test_clean_qwen25vl.jsonl 

整体结果：

text n: 1200 accuracy: 0.8758 prediction_frequency: {'A': 306, 'B': 317, 'C': 288, 'D': 289}  accuracy_by_positive_position: A 265 / 300 = 0.8833 B 270 / 300 = 0.9000 C 260 / 300 = 0.8667 D 256 / 300 = 0.8533  num_bases: 300 all_4_correct_consistency: 0.8300 any_correct: 0.9233 

Paper-style unstable-only analysis：

text total_bases: 300 all4_correct_bases: 249 unstable_bases: 51  Full-set position accuracy: A 265 / 300 = 0.8833 B 270 / 300 = 0.9000 C 260 / 300 = 0.8667 D 256 / 300 = 0.8533  full_position_gap: 0.0467  Paper-style unstable-only position accuracy: A 16 / 51 = 0.3137 B 21 / 51 = 0.4118 C 11 / 51 = 0.2157 D 7 / 51 = 0.1373  unstable_position_gap: 0.2745 

结论：

Balanced Clean 下 full-set gap 为 4.67%，unstable-only gap 达到 27.45%。这说明即使消除了 negative order confound，困难样本中仍然存在明显的 performance-level position sensitivity。

---

### 2. Balanced 4-position Occlusion

输出文件：

text result/caption_semvis_hard_large_balanced_test_occlusion_s2_qwen25vl.jsonl 

整体结果：

text n: 1200 accuracy: 0.8533 prediction_frequency: {'A': 307, 'B': 316, 'C': 286, 'D': 291}  accuracy_by_positive_position: A 261 / 300 = 0.8700 B 262 / 300 = 0.8733 C 249 / 300 = 0.8300 D 252 / 300 = 0.8400  num_bases: 300 all_4_correct_consistency: 0.7933 any_correct: 0.9133 

Paper-style unstable-only analysis：

text total_bases: 300 all4_correct_bases: 238 unstable_bases: 62  Full-set position accuracy: A 261 / 300 = 0.8700 B 262 / 300 = 0.8733 C 249 / 300 = 0.8300 D 252 / 300 = 0.8400  full_position_gap: 0.0433  Paper-style unstable-only position accuracy: A 23 / 62 = 0.3710 B 24 / 62 = 0.3871 C 11 / 62 = 0.1774 D 14 / 62 = 0.2258  unstable_position_gap: 0.2097 

结论：

Occlusion 使准确率从 87.58% 下降到 85.33%，all-4 consistency 从 83.00% 下降到 79.33%，unstable bases 从 51 增加到 62。说明视觉证据被遮挡后，模型在不同排列下更难保持稳定判断。即使在 balanced setting 下，occlusion 的 unstable-only gap 仍达到 20.97%，说明顺序敏感性并非仅由负样本固定顺序导致。

---

### 3. RAG-style Multi-image VQA Clean

输出文件：

text result/rag_vqa_test_clean_qwen25vl.jsonl 

整体结果：

text n: 800 accuracy: 0.4825 prediction_frequency: {'A': 165, 'B': 192, 'D': 199, 'C': 244}  accuracy_by_positive_position: Image 1 146 / 200 = 0.7300 Image 2 83 / 200 = 0.4150 Image 3 76 / 200 = 0.3800 Image 4 81 / 200 = 0.4050  num_bases: 200 all_4_correct_consistency: 0.3250 any_correct: 0.7500 

Paper-style unstable-only analysis：

text total_bases: 200 all4_correct_bases: 65 unstable_bases: 135  Full-set position accuracy: Image 1 146 / 200 = 0.7300 Image 2 83 / 200 = 0.4150 Image 3 76 / 200 = 0.3800 Image 4 81 / 200 = 0.4050  full_position_gap: 0.3500  Paper-style unstable-only position accuracy: Image 1 81 / 135 = 0.6000 Image 2 18 / 135 = 0.1333 Image 3 11 / 135 = 0.0815 Image 4 16 / 135 = 0.1185  unstable_position_gap: 0.5185 

结论：

RAG-style VQA 中出现了非常明显的 first-image advantage。当 relevant image 位于 Image 1 时，准确率达到 73.00%；而位于 Image 2、Image 3、Image 4 时，准确率分别下降到 41.50%、38.00%、40.50%。全样本 position gap 达到 35.00%，unstable-only gap 达到 51.85%。

这组结果最贴近原论文 RAG-VQA 的设置：问题和文本答案选项保持不变，只改变 question-relevant image 的位置。实验说明，Qwen2.5-VL-7B 在多图 VQA 中并非只是存在固定文本选项输出偏好，而是关键视觉证据图的位置会显著影响模型能否正确作答。

---

## 新增结果汇总表

| Dataset / Setting | Accuracy | All-4 Consistency | Unstable Bases | Full-set Gap | Unstable-only Gap |
|---|---:|---:|---:|---:|---:|
| Balanced Semantic+Category Clean | 87.58% | 83.00% | 51 / 300 | 4.67% | 27.45% |
| Balanced Semantic+Category Occlusion | 85.33% | 79.33% | 62 / 300 | 4.33% | 20.97% |
| RAG-style VQA Clean | 48.25% | 32.50% | 135 / 200 | 35.00% | 51.85% |

---

## 对当前实验主线的更新

当前实验可以按如下逻辑组织：

1. Object VQA / Caption Matching：基础复现实验，说明简单任务中 Qwen2.5-VL-7B 表现较稳定。
2. Semantic+Category HardNeg：提升候选图难度，使顺序敏感性更明显。
3. Balanced 4-position Semantic+Category HardNeg：排除原始 4-position 中 negative order confound，验证困难样本中仍存在 performance-level position sensitivity。
4. Occlusion：加入视觉扰动，说明视觉证据退化会进一步降低跨排列一致性。
5. RAG-style Multi-image VQA：更贴近原论文 RAG-VQA 设置，显示 relevant image position 对文本答案准确率有显著影响，尤其存在明显 first-image advantage。

最终可以强调：

text 模型没有表现出强烈固定文本选项输出偏好，但在困难多图任务中存在显著 order sensitivity。尤其在 RAG-style VQA 中，关键证据图位于 Image 1 时模型表现显著更好，说明模型对多图上下文中的视觉证据位置高度敏感。 

---

## 后续建议

1. 将 Balanced Clean / Balanced Occlusion / RAG-VQA Clean 作为当前主结果写入报告。
2. RAG-VQA 结果已经很强，200 base samples 足够作为补充实验；如果时间紧，不必继续扩大到 500 base samples。
3. 可以挑选 RAG-VQA 的 failure cases，展示同一 base sample 在 Image 1 正确、Image 2/3/4 错误的例子。
4. 后续方法部分建议强调 permutation-level robust aggregation，而不是只做固定 A/B/C/D position calibration。
