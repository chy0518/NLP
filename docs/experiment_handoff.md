OrderGuard / Order Sensitivity Experiment Handoff

本项目围绕多模态大模型在多图视觉问答与多图 caption matching 中的输入顺序敏感性展开。核心问题是：同一组图片与同一个问题，仅改变正确图片在 A/B/C/D 中的位置，模型输出是否稳定？

当前实验使用的主模型是：

/root/models/Qwen2.5-VL-7B-Instruct

服务器项目目录：

/root/autodl-tmp/OrderGuard/NLP

数据目录通过软链接接入：

/root/autodl-tmp/OrderGuard/NLP/data -> /root/autodl-tmp/OrderGuard/data

⸻

1. 已完成的主要工作

1.1 COCO 数据下载与基础数据构建

已在本地完成 COCO val2017 下载与解压：

data/coco/val2017/                              # 5000 张 COCO val2017 图片
data/coco/annotations/instances_val2017.json
data/coco/annotations/captions_val2017.json

基础 object-level VQA 数据集已构建：

data/orderguard_dev.jsonl
data/orderguard_test.jsonl

行数：

400  data/orderguard_dev.jsonl
1200 data/orderguard_test.jsonl

构造方式：

* 每个 base sample 包含 4 张图片；
* 其中 1 张 positive image 包含目标 COCO category；
* 另外 3 张 negative images 不包含该 category；
* 问题格式为：

Which image contains a [category]?

每个 base sample 展开为 4 个 positioned versions，使正确图分别出现在 A/B/C/D。

⸻

1.2 Caption Matching Random Negative Set

已构建随机负样本 caption matching 数据集：

data/orderguard_caption_hard_dev.jsonl
data/orderguard_caption_hard_test.jsonl

行数：

120 data/orderguard_caption_hard_dev.jsonl
400 data/orderguard_caption_hard_test.jsonl

构造方式：

* positive image 是 COCO caption 对应原图；
* negative images 是随机采样的其他图片；
* 问题格式：

Which image best matches the caption: "[caption]"?

实验发现该数据集对 Qwen2.5-VL-7B 过于简单，模型在 test clean 上达到 100% accuracy，因此后续不作为主结果。

⸻

1.3 Caption Hard Negative Set

为提高难度，构建了基于 COCO object category overlap 的 hard-negative caption matching 数据集：

data/orderguard_caption_hardneg_dev.jsonl
data/orderguard_caption_hardneg_test.jsonl

行数：

120 data/orderguard_caption_hardneg_dev.jsonl
400 data/orderguard_caption_hardneg_test.jsonl

构造统计：

Usable positive samples: 3927
Dev base samples: 30
Test base samples: 100
Dev rows: 120
Test rows: 400
Average shared category count per negative: 2.42

构造规则：

* positive image 是 caption 对应图片；
* negative images 与 positive image 至少共享 1 个 COCO object category；
* 优先选择 category overlap 更高的 hard negatives；
* 每个 base sample 展开为 A/B/C/D 四个位置版本。

⸻

1.4 Semantic + Category Hard Negative Set

进一步构建了 semantic-category hard negative caption matching 数据集：

data/orderguard_caption_semvis_hard_dev.jsonl
data/orderguard_caption_semvis_hard_test.jsonl

注意：当前版本是降级版，使用 --disable_image_similarity，因此实际是：

caption semantic similarity + COCO category overlap

不是完整 semantic-visual version。

行数：

120 data/orderguard_caption_semvis_hard_dev.jsonl
400 data/orderguard_caption_semvis_hard_test.jsonl

构造统计：

Usable positive samples: 3927
Dev rows: 120
Test rows: 400
Average shared category count per negative: 2.22
Average caption similarity: 0.4986
Average image similarity: 0.0000
Average category overlap score: 0.5632
Average final hard negative score: 0.3370

构造规则：

* positive image 是 caption 对应图片；
* negative images 需要同时满足：
    * 与 positive image 共享 COCO category；
    * 与 positive caption 的语义相似度较高；
* caption embedding 使用：

sentence-transformers/all-MiniLM-L6-v2

caption embedding 缓存路径：

data/cache/caption_embeddings_semvis.pkl

⸻

2. 服务器端已实现脚本

代码位于 Git 仓库：

/root/autodl-tmp/OrderGuard/NLP

主要脚本：

methods/corruption.py
eval/run_qwen_vl_eval.py
eval/analyze_outputs.py
eval/analyze_position_gap_paper_style.py
eval/find_failure_cases.py

2.1 图像扰动脚本

methods/corruption.py

支持：

clean
blur
noise
occlusion
brightness

当前主要使用：

blur s2
noise s2
occlusion s2

扰动后的图片会保存到：

data/corrupted/

例如：

data/corrupted/blur_s2_all_images/
data/corrupted/occlusion_s2_all_images/

⸻

2.2 Qwen2.5-VL 推理脚本

eval/run_qwen_vl_eval.py

功能：

* 读取 JSONL 数据；
* 每条样本输入 4 张图片 + 1 个问题；
* 调用 Qwen2.5-VL-7B-Instruct；
* 要求模型输出：

Answer: <A/B/C/D>
Evidence: <one short visual reason>

输出 JSONL 中包含：

sample_id
base_id
question
answer
prediction
is_correct
raw_response
options
positive_position
corruption
severity

运行示例：

cd /root/autodl-tmp/OrderGuard/NLP
MODEL=/root/models/Qwen2.5-VL-7B-Instruct
PYTHONPATH=. python eval/run_qwen_vl_eval.py \
  --model_path $MODEL \
  --input_jsonl data/orderguard_test.jsonl \
  --output_jsonl result/test_clean_qwen25vl.jsonl \
  --project_root . \
  --corruption clean \
  --max_new_tokens 64

⸻

2.3 常规分析脚本

eval/analyze_outputs.py

输出：

* overall accuracy；
* prediction frequency；
* accuracy by positive_position；
* number of base samples；
* all-4 correct consistency；
* any-correct ratio。

运行示例：

PYTHONPATH=. python eval/analyze_outputs.py \
  --input_jsonl result/test_clean_qwen25vl.jsonl

⸻

2.4 Paper-style unstable-only 分析脚本

eval/analyze_position_gap_paper_style.py

目的：

全样本 position gap 会被 easy samples 稀释。为了更接近原论文分析方式，该脚本会排除四个位置全部答对的 base samples，只分析真正受顺序影响的 unstable samples。

输出：

* total_bases；
* all4_correct_bases；
* unstable_bases；
* full-set position accuracy；
* full_position_gap；
* unstable-only position accuracy；
* unstable_position_gap。

运行示例：

PYTHONPATH=. python eval/analyze_position_gap_paper_style.py \
  --input_jsonl result/caption_semvis_hard_test_clean_qwen25vl.jsonl

⸻

2.5 Failure Case 查找脚本

eval/find_failure_cases.py

目的：

找同一个 base sample 的四个位置版本中，有的答对、有的答错的案例，用于 PPT 展示。

运行示例：

PYTHONPATH=. python eval/find_failure_cases.py \
  --input_jsonl result/caption_semvis_hard_test_clean_qwen25vl.jsonl \
  --top_k 5

⸻

3. 已完成实验结果

3.1 Object VQA Clean

输入：

data/orderguard_test.jsonl

输出：

result/test_clean_qwen25vl.jsonl

结果：

n: 1200
accuracy: 0.9575
prediction_frequency: {'A': 296, 'B': 307, 'C': 302, 'D': 295}
accuracy_by_positive_position:
A 287 / 300 = 0.9567
B 291 / 300 = 0.9700
C 288 / 300 = 0.9600
D 283 / 300 = 0.9433
num_bases: 300
all_4_correct_consistency: 0.9233
any_correct: 0.9800

结论：

Object-level VQA 对 Qwen2.5-VL-7B 较简单，模型准确率高，预测分布基本均衡，顺序敏感性不强。

⸻

3.2 Object VQA + Corruption

Blur s2

result/test_blur_s2_qwen25vl.jsonl
accuracy: 0.9308
prediction_frequency: {'A': 283, 'B': 305, 'C': 307, 'D': 305}
A 274 / 300 = 0.9133
B 284 / 300 = 0.9467
C 281 / 300 = 0.9367
D 278 / 300 = 0.9267
all_4_correct_consistency: 0.8833
any_correct: 0.9633

Noise s2

result/test_noise_s2_qwen25vl.jsonl
accuracy: 0.9333
prediction_frequency: {'A': 292, 'B': 309, 'C': 301, 'D': 298}
A 279 / 300 = 0.9300
B 284 / 300 = 0.9467
C 279 / 300 = 0.9300
D 278 / 300 = 0.9267
all_4_correct_consistency: 0.8867
any_correct: 0.9633

Occlusion s2

result/test_occlusion_s2_qwen25vl.jsonl
accuracy: 0.9333
prediction_frequency: {'A': 292, 'B': 309, 'C': 299, 'D': 300}
A 276 / 300 = 0.9200
B 285 / 300 = 0.9500
C 281 / 300 = 0.9367
D 278 / 300 = 0.9267
all_4_correct_consistency: 0.8867
any_correct: 0.9700

结论：

视觉扰动会降低整体准确率和 all-4 consistency，但在 object VQA 中还没有形成特别强的位置偏置。

⸻

3.3 Random Caption Matching

输入：

data/orderguard_caption_hard_test.jsonl

输出：

result/caption_hard_test_clean_qwen25vl.jsonl

结果：

accuracy: 1.0000
prediction_frequency: {'A': 100, 'B': 100, 'C': 100, 'D': 100}
all_4_correct_consistency: 1.0000

结论：

随机负样本 caption matching 对 Qwen2.5-VL-7B 过于简单，不适合作为主实验。

⸻

3.4 Caption HardNeg Clean

输入：

data/orderguard_caption_hardneg_test.jsonl

输出：

result/caption_hardneg_test_clean_qwen25vl.jsonl

结果：

n: 400
accuracy: 0.9300
prediction_frequency: {'A': 103, 'B': 101, 'C': 96, 'D': 100}
A 94 / 100 = 0.9400
B 95 / 100 = 0.9500
C 92 / 100 = 0.9200
D 91 / 100 = 0.9100
num_bases: 100
all_4_correct_consistency: 0.9000
any_correct: 0.9600

结论：

共享 COCO category 的 hard negatives 开始降低模型稳定性。

⸻

3.5 Caption HardNeg + Occlusion

输出：

result/caption_hardneg_test_occlusion_s2_qwen25vl.jsonl

结果：

n: 400
accuracy: 0.9200
prediction_frequency: {'A': 103, 'B': 101, 'C': 98, 'D': 98}
A 92 / 100 = 0.9200
B 94 / 100 = 0.9400
C 92 / 100 = 0.9200
D 90 / 100 = 0.9000
num_bases: 100
all_4_correct_consistency: 0.8700
any_correct: 0.9700

结论：

在 hard-negative caption matching 下，遮挡进一步降低跨位置一致性。

⸻

3.6 Semantic + Category HardNeg Clean

输入：

data/orderguard_caption_semvis_hard_test.jsonl

输出：

result/caption_semvis_hard_test_clean_qwen25vl.jsonl

结果：

n: 400
accuracy: 0.8575
prediction_frequency: {'A': 120, 'B': 98, 'C': 88, 'D': 94}
A 89 / 100 = 0.8900
B 84 / 100 = 0.8400
C 84 / 100 = 0.8400
D 86 / 100 = 0.8600
num_bases: 100
all_4_correct_consistency: 0.8000
any_correct: 0.9100

结论：

这是当前最关键的 clean setting。候选图在语义与类别层面都更相似后，模型准确率明显下降，并出现 A 选项偏好。

⸻

3.7 Semantic + Category HardNeg + Blur

输出：

result/caption_semvis_hard_test_blur_s2_qwen25vl.jsonl

结果：

n: 400
accuracy: 0.8475
prediction_frequency: {'A': 114, 'B': 106, 'C': 93, 'D': 87}
A 86 / 100 = 0.8600
B 86 / 100 = 0.8600
C 84 / 100 = 0.8400
D 83 / 100 = 0.8300
num_bases: 100
all_4_correct_consistency: 0.7800
any_correct: 0.9100

⸻

3.8 Semantic + Category HardNeg + Occlusion

输出：

result/caption_semvis_hard_test_occlusion_s2_qwen25vl.jsonl

结果：

n: 400
accuracy: 0.8425
prediction_frequency: {'A': 117, 'B': 105, 'C': 89, 'D': 89}
A 86 / 100 = 0.8600
B 86 / 100 = 0.8600
C 82 / 100 = 0.8200
D 83 / 100 = 0.8300
num_bases: 100
all_4_correct_consistency: 0.7700
any_correct: 0.9100

结论：

在最难数据集上加入视觉扰动后，all-4 consistency 进一步下降到 77%～78%，说明视觉证据退化会进一步削弱跨顺序稳定性。

⸻

4. Paper-style Unstable-only Analysis

由于全样本中包含大量 easy samples，直接计算 full-set position gap 会被四个位置全答对的样本稀释。参考原论文分析顺序敏感样本的思路，我们进一步排除 all-4-correct base samples，只分析真正受到顺序影响的 unstable samples。

Semantic + Category HardNeg Clean

输入：

result/caption_semvis_hard_test_clean_qwen25vl.jsonl

结果：

total_bases: 100
all4_correct_bases: 80
unstable_bases: 20
Full-set position accuracy:
A 89 / 100 = 0.8900
B 84 / 100 = 0.8400
C 84 / 100 = 0.8400
D 86 / 100 = 0.8600
full_position_gap: 0.0500
Paper-style unstable-only position accuracy:
A 9 / 20 = 0.4500
B 4 / 20 = 0.2000
C 4 / 20 = 0.2000
D 6 / 20 = 0.3000
unstable_position_gap: 0.2500

Semantic + Category HardNeg Occlusion

total_bases: 100
all4_correct_bases: 77
unstable_bases: 23
Full-set position accuracy:
A 86 / 100 = 0.8600
B 86 / 100 = 0.8600
C 82 / 100 = 0.8200
D 83 / 100 = 0.8300
full_position_gap: 0.0400
Paper-style unstable-only position accuracy:
A 9 / 23 = 0.3913
B 9 / 23 = 0.3913
C 5 / 23 = 0.2174
D 6 / 23 = 0.2609
unstable_position_gap: 0.1739

Semantic + Category HardNeg Blur

total_bases: 100
all4_correct_bases: 78
unstable_bases: 22
Full-set position accuracy:
A 86 / 100 = 0.8600
B 86 / 100 = 0.8600
C 84 / 100 = 0.8400
D 83 / 100 = 0.8300
full_position_gap: 0.0300
Paper-style unstable-only position accuracy:
A 8 / 22 = 0.3636
B 8 / 22 = 0.3636
C 6 / 22 = 0.2727
D 5 / 22 = 0.2273
unstable_position_gap: 0.1364

指标	数值
Total Base Samples	300
All-4 Correct Bases	244
Unstable Bases	56

全样本位置准确率：

Position	Correct / Total	Accuracy
A	265 / 300	88.33%
B	262 / 300	87.33%
C	256 / 300	85.33%
D	256 / 300	85.33%

Unstable-only 位置准确率：

Position	Correct / Unstable Bases	Accuracy
A	21 / 56	37.50%
B	18 / 56	32.14%
C	12 / 56	21.43%
D	12 / 56	21.43%

Unstable-only position gap：

37.50% - 21.43% = 16.07%

n: 1200

accuracy: 0.8592

prediction_frequency: {'A': 312, 'B': 317, 'C': 293, 'D': 278}

accuracy_by_positive_position:

A 261 / 300 = 0.8700

B 261 / 300 = 0.8700

C 257 / 300 = 0.8567

D 252 / 300 = 0.8400

num_bases: 300

all_4_correct_consistency: 0.8033

any_correct: 0.9067

all4_correct_bases: 244

all4_correct_bases: 241

unstable_bases: 59

Full-set gap: 0.0300

Unstable-only:

A 20 / 59 = 0.3390

B 20 / 59 = 0.3390

C 16 / 59 = 0.2712

D 11 / 59 = 0.1864

unstable_position_gap: 0.1525

⸻

5. 代表性失败案例

Case 1: Tennis Racket

Caption:

A man on a court with a tennis racket.

Base ID:

caption_semvis_hard_test_base_000015

结果：

A: correct
B: wrong, model predicts A
C: wrong, model predicts A
D: correct

解释：

负样本也包含 tennis court / person / racket 等强视觉线索。模型在 B/C 位置变体中被前面的 hard negative 吸引，输出 A。

⸻

Case 2: Three-course Dinner

Caption:

three course dinner is served on a table

Base ID:

caption_semvis_hard_test_base_000036

clean setting 中：

A: wrong
B: wrong
C: wrong
D: correct

blur setting 中：

A: correct
B: wrong
C: wrong
D: correct

解释：

多个候选图都含有 plates / food / table 等局部匹配线索，模型容易选择语义相近但非 COCO caption 对应的 hard negative。

⸻

Case 3: Lifeguard / Surfboard

Caption:

A lifeguard helps a boarder that overturned in the water.

Base ID:

caption_semvis_hard_test_base_000010

clean setting 中：

A: correct
B: correct
C: wrong
D: correct

blur setting 中：

A: wrong
B: correct
C: correct
D: correct

occlusion setting 中：

A: wrong
B: correct
C: wrong
D: wrong

解释：

该样本涉及动作关系和事件理解，不只是检测 surfboard。视觉扰动后，模型更容易把“person + surfboard + water”的相似干扰图当作答案。

⸻

6. 当前结论

当前实验支持以下结论：

1. 简单 object-level VQA 中，Qwen2.5-VL-7B 视觉识别能力较强，输入顺序影响不明显。
2. 视觉扰动会降低模型跨位置一致性。
3. 当候选图片共享物体类别，并且 caption 语义相近时，模型对输入顺序的鲁棒性明显下降。
4. 在 Semantic + Category HardNeg setting 中，模型出现较明显的 A 选项偏好。
5. 全样本 gap 会被 easy samples 稀释；在 unstable-only 分析中，position gap 可达到 13.64%～25.00%。

推荐报告中的核心表述：

在简单 object-level VQA 中，Qwen2.5-VL-7B 准确率达到 95.75%，all-4 consistency 为 92.33%，说明清晰视觉证据能够一定程度上压制顺序敏感性。随着任务升级为 Semantic + Category Hard Negative Caption Matching，候选图在类别和语义上更相似，模型准确率下降到 85.75%，all-4 consistency 下降到 80.00%，并出现对 A 选项的预测偏好。进一步加入 blur 和 occlusion 后，all-4 consistency 下降至 78.00% 和 77.00%。在 unstable-only analysis 中，位置 gap 达到 13.64%～25.00%，说明真正受顺序影响的样本上，模型存在明显的位置敏感性。

⸻

7. 需要继续做的事情

7.1 扩大 Semantic + Category HardNeg 数据集

目前 paper-style unstable-only 样本数只有 20～23 个，建议扩大最关键的数据集。

推荐规模：

dev: 50 base samples -> 200 rows
test: 300 base samples -> 1200 rows

目标文件：

data/orderguard_caption_semvis_hard_large_dev.jsonl
data/orderguard_caption_semvis_hard_large_test.jsonl

建议给构建脚本增加参数：

--output_prefix

运行目标命令：

python data/coco_caption_semantic_visual_hard_builder.py \
  --coco_root data/coco \
  --caption_file data/coco/annotations/captions_val2017.json \
  --instance_file data/coco/annotations/instances_val2017.json \
  --out_dir data \
  --output_prefix orderguard_caption_semvis_hard_large \
  --num_dev 50 \
  --num_test 300 \
  --seed 20260528 \
  --top_k_hard_negatives 50 \
  --disable_image_similarity

预期输出：

200  data/orderguard_caption_semvis_hard_large_dev.jsonl
1200 data/orderguard_caption_semvis_hard_large_test.jsonl

服务器上传：

rsync -avP -e "ssh -p 20464" \
  "/Users/ziyues/Documents/COCO Dataset/OrderGuard/data/orderguard_caption_semvis_hard_large_dev.jsonl" \
  "/Users/ziyues/Documents/COCO Dataset/OrderGuard/data/orderguard_caption_semvis_hard_large_test.jsonl" \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/OrderGuard/data/

服务器运行：

cd /root/autodl-tmp/OrderGuard/NLP
MODEL=/root/models/Qwen2.5-VL-7B-Instruct
PYTHONPATH=. python eval/run_qwen_vl_eval.py \
  --model_path $MODEL \
  --input_jsonl data/orderguard_caption_semvis_hard_large_test.jsonl \
  --output_jsonl result/caption_semvis_hard_large_test_clean_qwen25vl.jsonl \
  --project_root . \
  --corruption clean \
  --max_new_tokens 64
PYTHONPATH=. python eval/analyze_outputs.py \
  --input_jsonl result/caption_semvis_hard_large_test_clean_qwen25vl.jsonl
PYTHONPATH=. python eval/analyze_position_gap_paper_style.py \
  --input_jsonl result/caption_semvis_hard_large_test_clean_qwen25vl.jsonl

如 clean 结果趋势稳定，再跑 occlusion：

PYTHONPATH=. python eval/run_qwen_vl_eval.py \
  --model_path $MODEL \
  --input_jsonl data/orderguard_caption_semvis_hard_large_test.jsonl \
  --output_jsonl result/caption_semvis_hard_large_test_occlusion_s2_qwen25vl.jsonl \
  --project_root . \
  --corruption occlusion \
  --severity 2 \
  --max_new_tokens 64
PYTHONPATH=. python eval/analyze_outputs.py \
  --input_jsonl result/caption_semvis_hard_large_test_occlusion_s2_qwen25vl.jsonl
PYTHONPATH=. python eval/analyze_position_gap_paper_style.py \
  --input_jsonl result/caption_semvis_hard_large_test_occlusion_s2_qwen25vl.jsonl

⸻

7.2 实现 OrderGuard++ 聚合方法

后续如果要从“现象复现”推进到“方法改进”，建议实现：

1. Single Order baseline；
2. Permutation Voting；
3. Position-Calibrated Voting；
4. Evidence-aware OrderGuard++。

当前 raw outputs 已经保存了：

prediction
raw_response
evidence
positive_position
options

可以基于同一个 base_id 的四个 positioned versions 做聚合。

⸻

7.3 生成图表

建议画三类图：

1. Accuracy comparison；
2. All-4 Consistency comparison；
3. Prediction Frequency A/B/C/D。

建议重点展示：

Object VQA Clean
Object VQA Occlusion
Caption HardNeg Clean
Semantic+Category Clean
Semantic+Category Blur
Semantic+Category Occlusion

⸻

8. Git 注意事项

当前 Git 仓库路径：

/root/autodl-tmp/OrderGuard/NLP

不要提交大数据与结果文件：

data/coco/
data/corrupted/
result/*.jsonl
*.jsonl

提交代码时只 add 代码文件，例如：

git add methods/__init__.py methods/corruption.py \
  eval/run_qwen_vl_eval.py \
  eval/analyze_outputs.py \
  eval/analyze_position_gap_paper_style.py \
  eval/find_failure_cases.py
git commit -m "Add OrderGuard evaluation and analysis scripts"
git push

注意不要使用：

git add .

因为 data 是本地软链接，且结果文件不应上传 GitHub。

⸻

9. 一句话总结

当前实验已经说明：顺序敏感性不是在所有多图任务中都同样明显；当视觉证据清晰、候选图容易区分时，Qwen2.5-VL-7B 较稳定。但当候选图在类别和语义上更相似、或者加入视觉扰动后，模型的跨位置一致性显著下降，并在 unstable samples 上表现出接近原论文量级的位置敏感性。