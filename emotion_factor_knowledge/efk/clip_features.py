from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .common import l2_normalize, read_jsonl
from .constants import EMOTIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract normalized CLIP features per emotion")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--limit-per-emotion", type=int)
    args = parser.parse_args()

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        dtype = torch.float32
    model = CLIPModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)
    records = read_jsonl(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for emotion in EMOTIONS:
        subset = [record for record in records if record["emotion"] == emotion]
        if args.limit_per_emotion:
            subset = subset[: args.limit_per_emotion]
        vectors: list[np.ndarray] = []
        valid_records: list[dict] = []
        for start in tqdm(range(0, len(subset), args.batch_size), desc=emotion):
            batch = subset[start : start + args.batch_size]
            images: list[Image.Image] = []
            accepted: list[dict] = []
            for record in batch:
                try:
                    with Image.open(record["image_path"]) as image:
                        images.append(image.convert("RGB"))
                    accepted.append(record)
                except Exception as exc:
                    print(f"WARN unreadable image {record['image_path']}: {exc}")
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = model.get_image_features(**inputs)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            vectors.append(features.cpu().numpy().astype(np.float32))
            valid_records.extend(accepted)
        if not vectors:
            raise RuntimeError(f"No readable images for {emotion}")
        embeddings = l2_normalize(np.concatenate(vectors, axis=0).astype(np.float32))
        np.savez_compressed(
            args.output_dir / f"{emotion}.npz",
            embeddings=embeddings,
            image_ids=np.asarray([r["image_id"] for r in valid_records]),
            image_paths=np.asarray([r["image_path"] for r in valid_records]),
        )
    metadata = {"model": args.model, "normalized": True, "emotions": list(EMOTIONS)}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
