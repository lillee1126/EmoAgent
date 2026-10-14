# EmoAgent

这是 EmoAgent 图像情绪编辑流程的推理复现项目。给定一张原图和一个目标情绪，流程会分析图像、检索情绪因素知识（EFK）、规划并执行多步编辑、验证编辑步骤，并根据 Critic 反馈进行优化。

本复现使用 FLUX.2-klein-9B 作为图像编辑后端，并使用项目中的 ResNet-50 替代论文中的 EmoVIT 进行自动情绪评估。

## 示例：一张原图，八种目标情绪

以下图片来自同一张原图的一次示例运行

| 原图 | 愉悦（Amusement） | 愤怒（Anger） | 敬畏（Awe） |
|---|---|---|---|
| ![原图](examples/eg/source.png) | ![愉悦](examples/eg/amusement.png) | ![愤怒](examples/eg/anger.png) | ![敬畏](examples/eg/awe.png) |

| 满足（Contentment） | 厌恶（Disgust） | 兴奋（Excitement） | 恐惧（Fear） | 悲伤（Sadness） |
|---|---|---|---|---|
| ![满足](examples/eg/contentment.png) | ![厌恶](examples/eg/disgust.png) | ![兴奋](examples/eg/excitement.png) | ![恐惧](examples/eg/fear.png) | ![悲伤](examples/eg/sadness.png) |

## 项目结构

```text
emoagent/             核心流程、提示词、模型与检索后端
  pipeline.py         图像编辑主流程
  prompts.py          提示词构建
  backends.py         模型和知识检索后端
  source_images.py    原图分析
run.py                单张图像推理入口
batch_run.py          批量推理与抽样入口
examples/eg/          上方展示的原图和八种情绪编辑图
docs/                 实现说明与复现指南
tests/                 本地单元测试
resnet50_emoset_classifier/  ResNet-50 训练、评估与推理代码
emotion_factor_knowledge/   EFK 构建、审计与检索代码
```

## 环境要求

- Python 3.10 或更高版本
- 支持 CUDA 的 PyTorch 环境和 NVIDIA GPU
- Hugging Face 模型访问权限：`Qwen/Qwen2.5-VL-7B-Instruct` 和 `black-forest-labs/FLUX.2-klein-9B`。FLUX 为受限模型，请先在 Hugging Face 接受模型条款。
- 单独准备好的 EFK 数据和 ResNet-50 checkpoint
- 火山方舟 API Key。检索使用的嵌入模型和 endpoint 从 EFK 索引清单中读取。

请根据本机 CUDA 环境安装匹配版本的 PyTorch 和 torchvision，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

请在 shell 中设置密钥，或复制 `.env.example` 并自行加载私有配置。不要将 API Key 或访问令牌提交到仓库。

```bash
export ARK_API_KEY="你的方舟 API Key"
export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export EMOAGENT_BASE_DIR="/path/to/emoagent-data"
```

`EMOAGENT_BASE_DIR` 指向外部数据目录，结构如下：

```text
emoagent-data/
  emotion_factor_knowledge/
  efk_runs/main/knowledge/nodes.jsonl
  efk_runs/main/index/
  resnet50_emoset_classifier/runs/resnet50_emoset/best.pt
```

也可以通过命令行参数分别指定各文件路径。请将数据集、模型权重、密钥和推理输出保存在源码目录之外。

## 数据集

本项目的 ResNet-50 训练与 EFK 构建使用 EmoSet-118K。可从 [EmoSet-118K 数据集下载页](https://www.dropbox.com/scl/fi/myue506itjfc06m7svdw6/EmoSet-118K.zip?dl=0&rlkey=7f3oyjkr6zyndf0gau7t140rv) 获取数据，并查看 [EmoSet 官方项目主页](https://vcc.tech/EmoSet) 了解数据格式与使用声明。下载并解压后，按训练说明将数据集根目录传给 `--dataset-root`。

EmoSet 官方声明该数据仅限非商业用途，并禁止再分发数据集或其衍生版本；使用时需引用原论文。请在下载和使用前阅读并遵守官方声明。
## ResNet-50 情绪分类器训练

仓库包含 EmoSet-118K 八分类 ResNet-50 的完整训练、验证、测试和单图推理代码，见 [`resnet50_emoset_classifier/README.md`](resnet50_emoset_classifier/README.md)。数据集和训练权重不包含在仓库中，需要按该指南单独准备。示例训练命令：

```bash
cd resnet50_emoset_classifier
python train.py --dataset-root /path/to/EmoSet-118K --epochs 30 --batch-size 128 --workers 8 --verify-paths
```

## EFK 情绪因素知识库构建

仓库包含从 EmoSet 清单、CLIP 特征、情绪内聚类、簇过滤、Qwen 总结，到因素树、TextNode、向量索引和审计的完整构建代码，见 [`emotion_factor_knowledge/README.md`](emotion_factor_knowledge/README.md)。数据集、ResNet checkpoint、模型文件和 API 密钥需单独准备。请按 EFK 指南中的阶段命令逐步构建。

## 单张图像推理

```bash
python run.py \
  --image /path/to/image.png \
  --target-emotion sadness \
  --num-plans 1 \
  --efk-project-dir "$EMOAGENT_BASE_DIR/emotion_factor_knowledge" \
  --efk-nodes "$EMOAGENT_BASE_DIR/efk_runs/main/knowledge/nodes.jsonl" \
  --efk-index "$EMOAGENT_BASE_DIR/efk_runs/main/index" \
  --resnet-checkpoint "$EMOAGENT_BASE_DIR/resnet50_emoset_classifier/runs/resnet50_emoset/best.pt" \
  --output-root "$EMOAGENT_BASE_DIR/emoagent_runs"
```

默认使用 `cuda:0` 运行 Qwen 和 ResNet，使用 `cuda:1` 运行 FLUX。可以通过 `--qwen-device`、`--resnet-device` 和 `--flux-device` 修改设备；显存不足时可以尝试 `--flux-cpu-offload`。

## 可复现性与局限

- 运行清单会记录图像 ID、目标情绪分配、中间决策、验证结果和最终输出。
- 固定随机种子可控制图像抽样和情绪分配；模型生成结果仍可能因硬件和依赖版本不同而变化。
- Critic 验收和 ResNet 分数是自动化参考信号，不能替代人工评价。
- `Emo-A_ResNet` 等情绪指标使用 ResNet 替代论文中的 EmoVIT。
- 使用或分发前，请分别确认模型和数据集许可。FLUX.2-klein-9B 需要单独申请访问权限并遵守其许可条款。

## 数据集

Yang, J., Huang, Q., Ding, T., Lischinski, D., Cohen-Or, D., & Huang, H. (2023).
[EmoSet: A Large-scale Visual Emotion Dataset with Rich Attributes](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_EmoSet_A_Large-scale_Visual_Emotion_Dataset_with_Rich_Attributes_ICCV_2023_paper.html).
*Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 20383–20394.
