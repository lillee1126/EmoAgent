from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("model response root must be a JSON object")
    return value


def prepare_source_image(source: Path, output: Path, size: int = 512) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"Source image not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = list(image.size)
        if image.size == (size, size):
            prepared = image.copy()
            operation = "none"
        else:
            prepared = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
            operation = "center_crop_and_resize"
        prepared.save(output, format="PNG")
    return {
        "input_path": str(source.resolve()),
        "prepared_path": str(output.resolve()),
        "original_size": original_size,
        "prepared_size": [size, size],
        "operation": operation,
    }


def copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def require_keys(value: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{context} is missing fields: {missing}")
