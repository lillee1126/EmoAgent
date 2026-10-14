from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet50

from emoset import EMOTIONS, build_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    _, transform = build_transforms()
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    values, indices = probabilities.topk(min(args.top_k, len(EMOTIONS)))
    result = [{"emotion": EMOTIONS[i], "probability": float(v)} for v, i in zip(values.cpu(), indices.cpu())]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

