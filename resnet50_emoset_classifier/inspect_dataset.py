from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Validate EmoSet JSON splits without requiring PyTorch")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--decode-samples", type=int, default=32)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    info = json.loads((root / "info.json").read_text(encoding="utf-8"))
    labels = info["label2idx"]
    report = {"root": str(root), "labels": labels, "splits": {}}
    for split in ("train", "val", "test"):
        records = json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
        counts = Counter(record[0] for record in records)
        missing = [record[1] for record in records if not (root / record[1]).is_file()]
        decode_errors = []
        for record in records[: args.decode_samples]:
            try:
                with Image.open(root / record[1]) as image:
                    image.verify()
            except Exception as exc:
                decode_errors.append({"path": record[1], "error": str(exc)})
        report["splits"][split] = {
            "samples": len(records),
            "class_counts": dict(counts),
            "missing_images": len(missing),
            "decode_errors_in_sample": decode_errors,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

