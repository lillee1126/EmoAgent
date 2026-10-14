from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import stable_id, write_jsonl
from .constants import EMOTIONS


def load_split(root: Path, split: str) -> list[dict[str, str]]:
    split_path = root / f"{split}.json"
    with split_path.open("r", encoding="utf-8") as stream:
        source = json.load(stream)
    records: list[dict[str, str]] = []
    for row in source:
        if not isinstance(row, list) or len(row) < 2:
            raise ValueError(f"Malformed row in {split_path}: {row!r}")
        emotion, relative_path = str(row[0]), str(row[1])
        if emotion not in EMOTIONS:
            raise ValueError(f"Unexpected emotion {emotion!r} in {split_path}")
        absolute_path = (root / relative_path).resolve()
        records.append(
            {
                "image_id": stable_id(relative_path),
                "emotion": emotion,
                "relative_path": relative_path.replace("\\", "/"),
                "image_path": str(absolute_path),
                "split": split,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified EmoSet image manifest")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for split in args.splits:
        for record in load_split(root, split):
            key = record["relative_path"]
            if key in seen:
                continue
            seen.add(key)
            if args.verify and not Path(record["image_path"]).is_file():
                raise FileNotFoundError(record["image_path"])
            records.append(record)
    write_jsonl(args.output, records)
    counts = {emotion: sum(r["emotion"] == emotion for r in records) for emotion in EMOTIONS}
    print(json.dumps({"images": len(records), "by_emotion": counts}, indent=2))


if __name__ == "__main__":
    main()
