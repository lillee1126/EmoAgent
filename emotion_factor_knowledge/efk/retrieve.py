from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .build_embeddings import ArkTextEmbedder, embed_sentence_transformers
from .common import l2_normalize, read_json, read_jsonl
from .constants import ELEMENT_TYPES, EMOTIONS


class ExactL2Retriever:
    def __init__(self, nodes_path: Path, index_dir: Path) -> None:
        self.nodes = read_jsonl(nodes_path)
        self.embeddings = np.load(index_dir / "embeddings.npy").astype(np.float32)
        self.manifest = read_json(index_dir / "embedding_manifest.json")
        node_ids = read_json(index_dir / "node_ids.json")
        if len(self.nodes) != len(self.embeddings) or node_ids != [n["node_id"] for n in self.nodes]:
            raise ValueError("nodes.jsonl and embedding index are out of sync")

    def encode_query(self, text: str, base_url: str | None = None) -> np.ndarray:
        backend, model = self.manifest["backend"], self.manifest["model"]
        if backend == "volcengine-ark":
            vector = ArkTextEmbedder(
                model,
                base_url or self.manifest.get("base_url"),
                max_retries=5,
            ).embed(text)
        elif backend == "sentence-transformers":
            vector = embed_sentence_transformers([text], model, 1)[0]
        else:
            raise ValueError(f"Unsupported backend {backend!r}")
        if self.manifest["normalized"]:
            vector = l2_normalize(vector)
        return vector.astype(np.float32)

    def search(
        self,
        query_vector: np.ndarray,
        emotion: str,
        element_type: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        indices = [
            index
            for index, node in enumerate(self.nodes)
            if node["emotion"] == emotion
            and (element_type is None or node["element_type"] == element_type)
        ]
        if not indices:
            return []
        matrix = self.embeddings[indices]
        distances = np.linalg.norm(matrix - query_vector[None, :], axis=1)
        order = np.argsort(distances)[:top_k]
        return [
            {**self.nodes[indices[int(position)]], "l2_distance": float(distances[position])}
            for position in order
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-5 EFK TextNodes by exact L2 distance")
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--query", required=True, help="Qwen-extracted source-image semantic text")
    parser.add_argument("--emotion", choices=EMOTIONS, required=True)
    parser.add_argument("--element-type", choices=ELEMENT_TYPES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--base-url")
    args = parser.parse_args()

    retriever = ExactL2Retriever(args.nodes, args.index_dir)
    query_vector = retriever.encode_query(args.query, args.base_url)
    result = retriever.search(query_vector, args.emotion, args.element_type, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
