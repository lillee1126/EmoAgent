from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .common import read_json, stable_id, write_json, write_jsonl
from .constants import ELEMENT_TYPES, EMOTIONS


def _merge_duplicate(nodes: list[dict[str, Any]], candidate: dict[str, Any], threshold: float) -> None:
    for existing in nodes:
        score = SequenceMatcher(
            None,
            existing["description"].lower(),
            candidate["description"].lower(),
        ).ratio()
        if score > threshold:
            existing["source_cluster_ids"].extend(candidate["source_cluster_ids"])
            existing["source_cluster_ids"] = sorted(set(existing["source_cluster_ids"]))
            existing["representative_images"].extend(candidate["representative_images"])
            existing["representative_images"] = list(
                dict.fromkeys(existing["representative_images"])
            )
            existing["cluster_size_total"] += candidate["cluster_size_total"]
            existing["confidence"] = max(existing["confidence"], candidate["confidence"])
            return
    nodes.append(candidate)


def build_nodes(summary_dir: Path, duplicate_threshold: float) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for emotion in EMOTIONS:
        grouped: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ELEMENT_TYPES}
        for path in sorted((summary_dir / emotion).glob("*_cluster_*.json")):
            if path.name.endswith(".failed.json"):
                continue
            record = read_json(path)
            for factor in record["factors"]:
                element_type = factor["element_type"]
                description = factor["description"].strip()
                candidate = {
                    "node_id": f"{emotion}_{element_type}_{stable_id(description.lower())}",
                    "text": description,
                    "description": description,
                    "metadata": {
                        "emotion": emotion,
                        "element_type": element_type,
                    },
                    "emotion": emotion,
                    "element_type": element_type,
                    "source_cluster_ids": [record["cluster_id"]],
                    "cluster_size_total": int(record["cluster_size"]),
                    "cluster_cohesion": float(record["cluster_cohesion"]),
                    "representative_images": record["sampled_image_paths"],
                    "evidence_count": int(factor["evidence_count"]),
                    "confidence": float(factor["confidence"]),
                    "summarizer": record["model"],
                }
                _merge_duplicate(grouped[element_type], candidate, duplicate_threshold)
        for element_type in ELEMENT_TYPES:
            nodes.extend(grouped[element_type])
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build factor trees and EmoAgent TextNodes")
    parser.add_argument("--summaries-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duplicate-threshold", type=float, default=0.90)
    args = parser.parse_args()
    nodes = build_nodes(args.summaries_dir, args.duplicate_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "nodes.jsonl", nodes)

    counts: dict[str, dict[str, int]] = {}
    for emotion in EMOTIONS:
        children = []
        counts[emotion] = {}
        for element_type in ELEMENT_TYPES:
            selected = [
                node
                for node in nodes
                if node["emotion"] == emotion and node["element_type"] == element_type
            ]
            counts[emotion][element_type] = len(selected)
            factor_dir = args.output_dir / "factors" / emotion
            write_json(factor_dir / f"{element_type}.json", selected)
            children.append(
                {
                    "name": element_type,
                    "value": sum(node["cluster_size_total"] for node in selected),
                    "children": [
                        {
                            "name": node["description"],
                            "value": node["cluster_size_total"],
                            "node_id": node["node_id"],
                        }
                        for node in selected
                    ],
                }
            )
        tree = {
            "name": emotion,
            "value": sum(child["value"] for child in children),
            "children": children,
        }
        write_json(args.output_dir / "factor_trees" / f"{emotion}.json", tree)
    manifest = {
        "schema_version": 1,
        "node_count": len(nodes),
        "duplicate_threshold": args.duplicate_threshold,
        "element_types": list(ELEMENT_TYPES),
        "counts": counts,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
