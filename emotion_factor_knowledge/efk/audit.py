from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_json, read_jsonl
from .constants import ELEMENT_TYPES, EMOTIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all EFK construction stages")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    nodes_path = args.run_dir / "knowledge" / "nodes.jsonl"
    nodes = read_jsonl(nodes_path) if nodes_path.is_file() else []
    report = {"emotions": {}, "node_count": len(nodes), "problems": []}
    for emotion in EMOTIONS:
        raw_path = args.run_dir / "raw_clusters" / f"{emotion}.json"
        filtered_path = args.run_dir / "filtered_clusters" / f"{emotion}.json"
        raw = read_json(raw_path) if raw_path.is_file() else None
        filtered = read_json(filtered_path) if filtered_path.is_file() else None
        summary_dir = args.run_dir / "qwen_summaries" / emotion
        summaries = [
            path
            for path in summary_dir.glob("*_cluster_*.json")
            if not path.name.endswith(".failed.json")
        ] if summary_dir.is_dir() else []
        failures = list(summary_dir.glob("*.failed.json")) if summary_dir.is_dir() else []
        no_consensus = sum(
            read_json(path).get("status") == "no_consensus" for path in summaries
        )
        by_type = {
            element_type: sum(
                node["emotion"] == emotion and node["element_type"] == element_type
                for node in nodes
            )
            for element_type in ELEMENT_TYPES
        }
        row = {
            "raw_clusters": raw["raw_cluster_count"] if raw else None,
            "kept_clusters": filtered["kept_count"] if filtered else None,
            "rejected_clusters": filtered["rejected_count"] if filtered else None,
            "qwen_summaries": len(summaries),
            "qwen_no_consensus": no_consensus,
            "qwen_failures": len(failures),
            "nodes_by_type": by_type,
        }
        report["emotions"][emotion] = row
        if filtered and len(summaries) + len(failures) != filtered["kept_count"]:
            report["problems"].append(f"{emotion}: not every retained cluster has a Qwen result")
        if failures:
            report["problems"].append(f"{emotion}: {len(failures)} Qwen summaries failed")
        if nodes and sum(by_type.values()) == 0:
            report["problems"].append(f"{emotion}: no knowledge nodes")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
