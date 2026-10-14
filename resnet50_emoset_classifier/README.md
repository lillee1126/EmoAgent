# EmoSet-118K ResNet-50 八分类复现

这是“复现任务书-EmoAgent”的任务 1 独立代码目录。代码直接读取 EmoSet 官方
`train.json`、`val.json`、`test.json`，不会改写数据集。

## 数据集下载

- [EmoSet-118K 数据集下载页](https://www.dropbox.com/scl/fi/myue506itjfc06m7svdw6/EmoSet-118K.zip?dl=0&rlkey=7f3oyjkr6zyndf0gau7t140rv)
- [EmoSet 官方项目主页](https://vcc.tech/EmoSet)

下载并解压后，使用 `--dataset-root /path/to/EmoSet-118K` 指定数据根目录。数据集官方声明仅限非商业用途且禁止再分发；请先阅读主页上的完整使用条款。

## 1. 环境

建议 Python 3.9、CUDA 12.x 与独立虚拟环境。PyTorch 请按服务器 CUDA 版本从官方源安装，
再安装其余依赖：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. 数据完整性检查

```powershell
python inspect_dataset.py "/path/to/EmoSet-118K"
```

## 3. 冒烟训练

先用每个 split 的前 256 个样本检查完整训练链路：

```powershell
python train.py --limit 256 --epochs 1 --workers 0 --batch-size 16 --output-dir runs/smoke
```

## 4. 正式训练

单卡：

```powershell
python train.py --epochs 30 --batch-size 128 --workers 8 --verify-paths
```

多卡（简单 DataParallel）：

```powershell
python train.py --epochs 30 --batch-size 256 --workers 12 --data-parallel
```

默认查找仓库根目录下的 `data/EmoSet-118K`；也可显式传入
`--dataset-root /path/to/EmoSet-118K`。最佳模型以验证集 macro-F1 选择，输出包括
`best.pt`、`last.pt`、训练历史、测试指标和混淆矩阵。

断点续训：

```powershell
python train.py --resume runs/resnet50_emoset/last.pt
```

## 5. 测试与单图推理

```powershell
python evaluate.py runs/resnet50_emoset/best.pt --split test
python infer.py runs/resnet50_emoset/best.pt path/to/image.jpg --top-k 3
```

类别顺序严格沿用 `info.json`：amusement、awe、contentment、excitement、anger、
disgust、fear、sadness。训练使用 ImageNet 预训练 ResNet-50、轻量数据增强、类别加权、
label smoothing、AdamW、CosineAnnealing、AMP 和早停。

## 6. 服务器训练

将数据集放到服务器并安装依赖后，可直接运行训练入口：

```bash
python train.py --dataset-root /path/to/EmoSet-118K --epochs 30 --batch-size 64 --workers 8 --output-dir runs/resnet50_emoset
```

所有权重、日志和指标保存在输出目录中；这些生成文件不属于源码包。


