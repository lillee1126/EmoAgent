from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .common import l2_normalize, write_json
from .constants import EMOTIONS


@dataclass
class ActiveCluster:
    members: list[int]
    centroid: np.ndarray
    version: int = 0


def _initial_edges(
    vectors: np.ndarray,
    threshold: float,
    block_size: int,
    device: torch.device,
) -> list[tuple[float, int, int, int, int]]:
    """Return every singleton pair above threshold without materializing NxN."""
    tensor = torch.from_numpy(vectors).to(device)
    heap: list[tuple[float, int, int, int, int]] = []
    total_blocks = (len(vectors) + block_size - 1) // block_size
    for left_block in tqdm(range(total_blocks), desc="pairwise blocks", leave=False):
        left_start = left_block * block_size
        left_end = min(left_start + block_size, len(vectors))
        left = tensor[left_start:left_end]
        for right_block in range(left_block, total_blocks):
            right_start = right_block * block_size
            right_end = min(right_start + block_size, len(vectors))
            similarities = (left @ tensor[right_start:right_end].T).float().cpu().numpy()
            rows, cols = np.where(similarities >= threshold)
            for row, col in zip(rows.tolist(), cols.tolist()):
                a, b = left_start + row, right_start + col
                if a >= b:
                    continue
                heap.append((-float(similarities[row, col]), a, b, 0, 0))
    heapq.heapify(heap)
    return heap


def greedy_centroid_clustering(
    vectors: np.ndarray,
    threshold: float = 0.89,
    block_size: int = 2048,
    device: str = "cuda",
) -> list[list[int]]:
    """EmoEdit-style greedy agglomeration using normalized mean centroids.

    The heap implementation is equivalent to repeatedly merging the currently
    most similar centroid pair above the appendix threshold, but avoids keeping
    the complete similarity matrix in RAM.
    """
    vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    clusters = {
        index: ActiveCluster([index], vectors[index].copy()) for index in range(len(vectors))
    }
    active: set[int] = set(clusters)
    heap = _initial_edges(vectors, threshold, block_size, torch_device)
    progress = tqdm(total=max(len(vectors) - 1, 0), desc="cluster merges", leave=False)
    while heap:
        negative_similarity, a, b, version_a, version_b = heapq.heappop(heap)
        if a not in active or b not in active:
            continue
        if clusters[a].version != version_a or clusters[b].version != version_b:
            continue
        similarity = -negative_similarity
        if similarity < threshold:
            break
        if len(clusters[a].members) < len(clusters[b].members):
            a, b = b, a
        left, right = clusters[a], clusters[b]
        size_left, size_right = len(left.members), len(right.members)
        centroid = (size_left * left.centroid + size_right * right.centroid) / (size_left + size_right)
        centroid = l2_normalize(centroid)
        left.members.extend(right.members)
        left.centroid = centroid.astype(np.float32)
        left.version += 1
        active.remove(b)
        del clusters[b]

        other_ids = [cluster_id for cluster_id in active if cluster_id != a]
        if other_ids:
            other_centroids = np.stack([clusters[cluster_id].centroid for cluster_id in other_ids])
            similarities = other_centroids @ left.centroid
            for other_id, score in zip(other_ids, similarities.tolist()):
                if score >= threshold:
                    first, second = sorted((a, other_id))
                    heapq.heappush(
                        heap,
                        (
                            -float(score),
                            first,
                            second,
                            clusters[first].version,
                            clusters[second].version,
                        ),
                    )
        progress.update(1)
    progress.close()
    return [sorted(cluster.members) for cluster in clusters.values()]


def cluster_emotion(feature_path: Path, emotion: str, args: argparse.Namespace) -> dict:
    data = np.load(feature_path, allow_pickle=False)
    vectors = data["embeddings"].astype(np.float32)
    image_ids = data["image_ids"].astype(str)
    image_paths = data["image_paths"].astype(str)
    groups = greedy_centroid_clustering(
        vectors,
        threshold=args.threshold,
        block_size=args.block_size,
        device=args.device,
    )
    records = []
    for cluster_number, member_indices in enumerate(sorted(groups, key=lambda x: (-len(x), x[0]))):
        member_vectors = vectors[member_indices]
        centroid = l2_normalize(member_vectors.mean(axis=0))
        cohesion = float(np.mean(member_vectors @ centroid))
        records.append(
            {
                "cluster_id": f"{emotion}_cluster_{cluster_number:05d}",
                "emotion": emotion,
                "size": len(member_indices),
                "cohesion": cohesion,
                "member_indices": member_indices,
                "image_ids": image_ids[member_indices].tolist(),
                "image_paths": image_paths[member_indices].tolist(),
            }
        )
    return {
        "emotion": emotion,
        "threshold": args.threshold,
        "feature_file": str(feature_path.resolve()),
        "raw_cluster_count": len(records),
        "clusters": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build appendix-style emotion-internal clusters")
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.89)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--emotions", nargs="+", choices=EMOTIONS, default=list(EMOTIONS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for emotion in args.emotions:
        result = cluster_emotion(args.features_dir / f"{emotion}.npz", emotion, args)
        write_json(args.output_dir / f"{emotion}.json", result)
        summary[emotion] = result["raw_cluster_count"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
