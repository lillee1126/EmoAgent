# EmoAgent 任务二：Emotion-Factor Knowledge Database

本目录实现任务书中的“利用 EmoSet 图像构建情感因素知识库”。实现优先遵循
EmoAgent 附录，再用 EmoEdit 论文及其公开源码补齐附录未说明的步骤。

数据来源： [EmoSet-118K 下载页](https://www.dropbox.com/scl/fi/myue506itjfc06m7svdw6/EmoSet-118K.zip?dl=0&rlkey=7f3oyjkr6zyndf0gau7t140rv) | [EmoSet 官方项目主页](https://vcc.tech/EmoSet)。本流程使用其 train/val/test 三个 split；数据集仅限非商业用途且禁止再分发，请遵守官方使用声明。

## 方法对应关系

| 实现 | 依据 |
|---|---|
| 八种情绪分别聚类 | EmoAgent 附录 / EmoEdit |
| CLIP 图像语义特征 | EmoAgent 附录 / EmoEdit |
| 每轮合并最相似的两个簇 | EmoAgent 附录 |
| 簇间相似度低于 `0.89` 停止 | EmoAgent 附录示例值 |
| 删除少于 5 张图的簇 | EmoAgent 附录 |
| 删除像素级过度重复的簇 | EmoAgent 附录；具体阈值未公开 |
| 删除不能明确唤起所属情绪的簇 | EmoAgent 附录；本实现使用任务一 ResNet-50 |
| Qwen2.5-VL-7B 总结簇 | 任务书指定，用于替代 GPT-4o/GPT-4V |
| object/scene/action/facial_expression/color_tone | EmoAgent 附录五个维度 |
| 每个保留簇随机抽 5 张图 | EmoEdit `GPT/code/Summary.py`；增加固定种子保证复现 |
| 相似描述字符串比率大于 `0.90` 合并 | EmoEdit `generate_json.py` |
| 因素转为带 emotion/element_type 标签的 TextNode | EmoAgent 附录 |
| 火山方舟 Doubao 多模态向量 API | 本复现替代不可用的 `text-embedding-ada-002` |
| 精确 L2 距离 Top-5 | EmoAgent 附录与正文 |

没有使用 FAISS 或 Chroma。知识节点数量较少，检索按照论文公式直接用 NumPy
计算 L2 距离。这样也避免把工程优化误写成论文方法。

## 目录与阶段产物

```text
EmoSet train/val/test JSON
  -> manifest.jsonl
  -> clip_features/{emotion}.npz
  -> raw_clusters/{emotion}.json
  -> filtered_clusters/{emotion}.json
  -> qwen_summaries/{emotion}/{cluster_id}.json
  -> knowledge/factor_trees/{emotion}.json
  -> knowledge/factors/{emotion}/{element_type}.json
  -> knowledge/nodes.jsonl
  -> index/embeddings.npy
```

各阶段都有独立、可审计的产物。Qwen 每完成一个簇就写一个 JSON，长任务中断后
不会丢失已经完成的总结结果。

## AutoDL 环境准备

代码目标环境为 Python 3.9。将本目录上传到：

```text
/root/autodl-tmp/emoagent/emotion_factor_knowledge
```

安装依赖：

```bash
cd /root/autodl-tmp/emoagent/emotion_factor_knowledge
pip install -r requirements.txt
```

建议提前下载：

```text
openai/clip-vit-large-patch14
Qwen/Qwen2.5-VL-7B-Instruct
```

ResNet-50 权重默认按任务一目录读取：

```text
/root/autodl-tmp/emoagent/resnet50_emoset_classifier/runs/resnet50_emoset/best.pt
```

只要 checkpoint 包含任务一保存的 `model` state dict 即可。

## 分阶段运行

以下命令均在本目录执行。

### 1. 汇总 EmoSet 三个官方 split

```bash
RUN=/root/autodl-tmp/emoagent/efk_runs/main
DATA=/root/autodl-tmp/nku-test/EmoSet-118K

python -m efk.manifest \
  --dataset-root "$DATA" \
  --output "$RUN/manifest.jsonl" \
  --verify
```

知识抽取使用全部 EmoSet 图像；推理评测使用独立的 EmoEdit-Inference-Set，因此不会与
405 张推理图产生重叠。若只希望使用训练集，可添加 `--splits train`。

### 2. 提取并缓存 CLIP 特征

```bash
python -m efk.clip_features \
  --manifest "$RUN/manifest.jsonl" \
  --output-dir "$RUN/clip_features" \
  --model openai/clip-vit-large-patch14 \
  --batch-size 64
```

特征写成每个情绪一个 NPZ，并已经 L2 归一化。

### 3. 情绪内部层次聚类

```bash
python -m efk.clustering \
  --features-dir "$RUN/clip_features" \
  --output-dir "$RUN/raw_clusters" \
  --threshold 0.89 \
  --block-size 2048
```

算法与附录一致：初始每图一簇，每轮合并当前余弦相似度最高的簇中心。实现使用堆和
分块矩阵乘法避免常驻完整 `N x N` 矩阵，但没有改变合并准则，也没有使用近似近邻。

### 4. 三步后处理

```bash
python -m efk.filter_clusters \
  --clusters-dir "$RUN/raw_clusters" \
  --output-dir "$RUN/filtered_clusters" \
  --resnet-checkpoint /root/autodl-tmp/emoagent/resnet50_emoset_classifier/runs/resnet50_emoset/best.pt \
  --score-cache "$RUN/resnet_scores.json" \
  --min-cluster-size 5 \
  --max-hamming-distance 2 \
  --min-unique-ratio 0.50 \
  --min-mean-emotion-score 0.30 \
  --min-target-hit-rate 0.50
```

论文没有公开后三个数值，本实现将其显式记录为可调工程参数。`rejected` 字段保存每个
被删除簇及原因，便于做阈值消融，绝不静默丢数据。

本地没有 ResNet 权重时，可以仅做管线测试：

```bash
python -m efk.filter_clusters \
  --clusters-dir "$RUN/raw_clusters" \
  --output-dir "$RUN/filtered_without_emotion_check" \
  --score-cache "$RUN/unused.json" \
  --skip-emotion-filter
```

正式结果不得使用该选项。

### 5. Qwen2.5-VL 总结每个保留簇

```bash
python -m efk.qwen_summarize \
  --filtered-dir "$RUN/filtered_clusters" \
  --output-dir "$RUN/qwen_summaries" \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --images-per-cluster 5 \
  --seed 2024
```

默认严格补齐 EmoEdit 公开源码行为：每个保留簇随机无放回抽 5 张。不同之处是固定了
抽样种子。Qwen 被要求只输出至少半数样本支持的因素，一个簇可按 EmoAgent 附录输出
多个维度，但每个维度最多一个因素。非法 JSON、非法类型、少数样本因素会自动重试；
连续失败会保存 `*.failed.json` 供人工复核。

### 6. 构建因素树和 TextNode

```bash
python -m efk.build_knowledge \
  --summaries-dir "$RUN/qwen_summaries" \
  --output-dir "$RUN/knowledge" \
  --duplicate-threshold 0.90
```

随后检查每个保留簇是否都有总结、每种情绪是否都有节点：

```bash
python -m efk.audit --run-dir "$RUN"
```

`nodes.jsonl` 的核心结构：

```json
{
  "node_id": "sadness_scene_...",
  "text": "a solitary person beneath an overcast sky",
  "metadata": {
    "emotion": "sadness",
    "element_type": "scene"
  },
  "source_cluster_ids": ["sadness_cluster_00012"],
  "representative_images": ["..."],
  "cluster_size_total": 21
}
```

这是可序列化的 EmoAgent TextNode 等价结构，同时保留簇和证据图像以供审计。

### 7. 使用火山方舟生成文本向量

```bash
export ARK_API_KEY=新创建的方舟API密钥
export ARK_EMBEDDING_MODEL=ep-20260802175844-r6z6w
export ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

python -m efk.build_embeddings \
  --nodes "$RUN/knowledge/nodes.jsonl" \
  --output-dir "$RUN/index" \
  --backend volcengine-ark \
  --model "$ARK_EMBEDDING_MODEL" \
  --base-url "$ARK_BASE_URL"
```

节点和查询必须使用同一个方舟接入点、同一输入类型和同一归一化设置。接口以
`[{"type":"text","text":"..."}]` 发送因素文本或 Qwen 提取的源图语义文本。
论文使用原始 L2，本实现默认不额外归一化。每个成功响应立即写入
`embedding_cache.jsonl`，中断后不会重复计费。使用豆包模型替代 ada-002 必须在报告中
明确标为模型替换，但 TextNode、元数据过滤和 L2 Top-5 逻辑保持不变。
正式构建前可先单独运行 `python -m efk.ark_preflight --model "$ARK_EMBEDDING_MODEL" --base-url "$ARK_BASE_URL"` 检查 API 密钥和 endpoint，避免长任务运行后才发现配置错误。

### 8. L2 Top-5 检索测试

```bash
python -m efk.retrieve \
  --nodes "$RUN/knowledge/nodes.jsonl" \
  --index-dir "$RUN/index" \
  --query "a child standing beside a bright lake under a clear sky" \
  --emotion sadness \
  --element-type scene \
  --top-k 5
```

任务三中，`--query` 应替换为 Qwen 从待编辑源图提取的语义文本。

## 按阶段构建

本目录提供下方各阶段的完整命令。依次运行清单生成、CLIP 特征、聚类、过滤、Qwen 总结、知识构建、审计和向量化即可完成构建。运行前先设置数据集、checkpoint、运行目录和 API 密钥环境变量。


最后的向量阶段需要环境变量 `ARK_API_KEY`，请勿将密钥写入配置文件或提交到代码仓库。每个阶段会将产物写入运行目录；遇到中断时，可从失败阶段继续运行。

## 验证

无需模型权重的单元测试：

```bash
python -m unittest discover -s tests -v
python -m compileall -q efk tests
```

正式运行后至少核对：

1. 八种情绪都有保留簇和因素节点；
2. `raw/kept/rejected/final node` 四组计数均已保存；
3. 随机抽查每种情绪至少 10 个簇的 5 张证据图和 Qwen 总结；
4. `*.failed.json` 数量为零或已人工处理；
5. 每种因素类型的 Top-5 检索不存在明显错序或大量同义重复；
6. 报告中明确区分论文参数、源码参数和本实现可调参数。

## Qwen 失败簇续跑

若流程在 Qwen 总结或后续阶段中断，重新运行上文的 Qwen 总结命令即可。程序会跳过已有的成功总结，只处理缺失或 `*.failed.json` 对应的簇；随后继续运行知识构建、审计和向量化命令。

校验按因子逐条进行：少于 3/5 图支持的因子被丢弃，不会令整个簇失败；同一
`element_type` 可以包含多个不同的具体因素；没有任何多数共识因素的簇保存为
`status: no_consensus`，该簇计入已处理数量但不会生成虚假知识节点。重试成功后，
旧的 `*.failed.json` 会被自动删除。




