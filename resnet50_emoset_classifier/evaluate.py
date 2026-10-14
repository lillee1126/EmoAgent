from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from emoset import EMOTIONS, EmoSetDataset, build_transforms, default_dataset_root
from engine import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    _, transform = build_transforms()
    dataset = EmoSetDataset(args.dataset_root, args.split, transform=transform)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    metrics, matrix = evaluate(model, loader, criterion, device, device.type == "cuda", len(EMOTIONS))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savetxt(args.output_dir / f"{args.split}_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

