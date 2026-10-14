from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

from .common import read_json, write_json
from .constants import EMOTIONS


def difference_hash(path: str) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel().tolist():
        value = (value << 1) | int(bit)
    return value


def unique_ratio(paths: Iterable[str], max_hamming_distance: int) -> float:
    hashes: list[int] = []
    total = 0
    for path in paths:
        total += 1
        current = difference_hash(path)
        if all(bin(current ^ previous).count("1") > max_hamming_distance for previous in hashes):
            hashes.append(current)
    return len(hashes) / max(total, 1)


class _ImageDataset(Dataset):
    def __init__(self, paths: list[str], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return path, tensor


def _build_resnet(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS))
    model.load_state_dict(state)
    return model.to(device).eval()


def predict_emotions(
    paths: list[str],
    checkpoint_path: Path,
    batch_size: int,
    workers: int,
    device_name: str,
) -> dict[str, list[float]]:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = _build_resnet(checkpoint_path, device)
    transform = transforms.Compose(
        [
            transforms.Resize(int(224 * 256 / 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    loader = DataLoader(
        _ImageDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    result: dict[str, list[float]] = {}
    for batch_paths, images in tqdm(loader, desc="ResNet-50 emotion scores"):
        with torch.inference_mode():
            probabilities = model(images.to(device)).softmax(dim=-1).cpu().numpy()
        for path, probability in zip(batch_paths, probabilities):
            result[str(path)] = probability.astype(float).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply EmoAgent appendix cluster filters")
    parser.add_argument("--clusters-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resnet-checkpoint", type=Path)
    parser.add_argument("--score-cache", type=Path, required=True)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--max-hamming-distance", type=int, default=2)
    parser.add_argument("--min-unique-ratio", type=float, default=0.50)
    parser.add_argument("--min-mean-emotion-score", type=float, default=0.30)
    parser.add_argument("--min-target-hit-rate", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-emotion-filter", action="store_true")
    parser.add_argument("--emotions", nargs="+", choices=EMOTIONS, default=list(EMOTIONS))
    args = parser.parse_args()

    raw_by_emotion = {
        emotion: read_json(args.clusters_dir / f"{emotion}.json") for emotion in args.emotions
    }
    paths = sorted(
        {
            path
            for value in raw_by_emotion.values()
            for cluster in value["clusters"]
            if cluster["size"] >= args.min_cluster_size
            for path in cluster["image_paths"]
        }
    )
    scores: dict[str, list[float]] = {}
    if not args.skip_emotion_filter:
        if args.score_cache.is_file():
            scores = read_json(args.score_cache)
        else:
            if not args.resnet_checkpoint:
                parser.error("--resnet-checkpoint is required unless --skip-emotion-filter is used")
            scores = predict_emotions(
                paths,
                args.resnet_checkpoint,
                args.batch_size,
                args.workers,
                args.device,
            )
            write_json(args.score_cache, scores)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall: dict[str, dict] = {}
    for emotion, source in raw_by_emotion.items():
        target_index = EMOTIONS.index(emotion)
        kept, rejected = [], []
        for cluster in tqdm(source["clusters"], desc=f"filter {emotion}"):
            reasons: list[str] = []
            metrics: dict[str, float] = {}
            if cluster["size"] < args.min_cluster_size:
                reasons.append("cluster_size")
            else:
                try:
                    metrics["unique_ratio"] = unique_ratio(
                        cluster["image_paths"], args.max_hamming_distance
                    )
                except Exception as exc:
                    reasons.append(f"pixel_filter_error:{type(exc).__name__}")
                else:
                    if metrics["unique_ratio"] < args.min_unique_ratio:
                        reasons.append("pixel_redundancy")

                if not args.skip_emotion_filter:
                    probabilities = np.asarray([scores[path] for path in cluster["image_paths"]])
                    metrics["mean_target_probability"] = float(probabilities[:, target_index].mean())
                    metrics["target_hit_rate"] = float(
                        np.mean(probabilities.argmax(axis=1) == target_index)
                    )
                    if metrics["mean_target_probability"] < args.min_mean_emotion_score:
                        reasons.append("low_emotion_score")
                    if metrics["target_hit_rate"] < args.min_target_hit_rate:
                        reasons.append("low_target_hit_rate")
            enriched = {**cluster, "filter_metrics": metrics}
            if reasons:
                rejected.append({**enriched, "rejection_reasons": reasons})
            else:
                kept.append(enriched)
        output = {
            "emotion": emotion,
            "source": str((args.clusters_dir / f"{emotion}.json").resolve()),
            "filter_config": {
                "min_cluster_size": args.min_cluster_size,
                "max_hamming_distance": args.max_hamming_distance,
                "min_unique_ratio": args.min_unique_ratio,
                "min_mean_emotion_score": None
                if args.skip_emotion_filter
                else args.min_mean_emotion_score,
                "min_target_hit_rate": None
                if args.skip_emotion_filter
                else args.min_target_hit_rate,
            },
            "kept_count": len(kept),
            "rejected_count": len(rejected),
            "clusters": kept,
            "rejected": rejected,
        }
        write_json(args.output_dir / f"{emotion}.json", output)
        overall[emotion] = {"kept": len(kept), "rejected": len(rejected)}
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
