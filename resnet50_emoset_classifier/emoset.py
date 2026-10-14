from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset


EMOTIONS = (
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
)


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "EmoSet-118K"


class EmoSetDataset(Dataset):
    """EmoSet loader for the official train/val/test JSON split files."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        transform: Optional[Callable] = None,
        verify_paths: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.transform = transform

        with (self.root / "info.json").open("r", encoding="utf-8") as stream:
            info = json.load(stream)
        self.label_to_idx = {str(k): int(v) for k, v in info["label2idx"].items()}
        self.idx_to_label = {int(k): str(v) for k, v in info["idx2label"].items()}
        if tuple(self.idx_to_label[i] for i in range(len(self.idx_to_label))) != EMOTIONS:
            raise ValueError("Unexpected EmoSet label mapping in info.json")

        with (self.root / f"{split}.json").open("r", encoding="utf-8") as stream:
            records = json.load(stream)
        if limit is not None:
            records = records[:limit]

        self.samples: list[tuple[Path, int]] = []
        missing: list[str] = []
        for record in records:
            if not isinstance(record, list) or len(record) < 2:
                raise ValueError(f"Malformed {split}.json record: {record!r}")
            label_name, relative_image_path = record[0], record[1]
            image_path = self.root / relative_image_path
            if verify_paths and not image_path.is_file():
                missing.append(str(image_path))
            self.samples.append((image_path, self.label_to_idx[label_name]))
        if missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(f"{len(missing)} image files are missing. First entries:\n{preview}")

        counts = Counter(label for _, label in self.samples)
        self.class_counts = [counts.get(i, 0) for i in range(len(self.label_to_idx))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int = 224):
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform

