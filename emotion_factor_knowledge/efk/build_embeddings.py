from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import ensure_parent, l2_normalize, read_jsonl, write_json


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def extract_ark_embedding(response: Any) -> np.ndarray:
    """Accept both documented Ark response shapes seen across SDK versions."""
    data = _field(response, "data")
    if data is None:
        raise ValueError("Ark response has no data field")
    if isinstance(data, (list, tuple)):
        if len(data) != 1:
            raise ValueError(f"Expected one embedding result, got {len(data)}")
        data = data[0]
    embedding = _field(data, "embedding")
    if embedding is None:
        raise ValueError("Ark response data has no embedding field")
    array = np.asarray(embedding, dtype=np.float32)
    # Some multimodal API/SDK versions wrap the single fused vector in one extra list.
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"Unexpected Ark embedding shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Ark embedding contains NaN or infinity")
    return array


class ArkTextEmbedder:
    def __init__(self, model: str, base_url: str | None, max_retries: int) -> None:
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError as exc:
            raise RuntimeError(
                'Install the Ark SDK with: pip install --upgrade "volcengine-python-sdk[ark]>=4.0.43"'
            ) from exc
        api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            raise RuntimeError("ARK_API_KEY is not set")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = Ark(**kwargs)
        self.model = model
        self.max_retries = max_retries

    def embed(self, text: str) -> np.ndarray:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.multimodal_embeddings.create(
                    model=self.model,
                    input=[{"type": "text", "text": text}],
                )
                return extract_ark_embedding(response)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = min(2**attempt, 30) + random.random()
                time.sleep(delay)
        raise RuntimeError(f"Ark embedding request failed after retries: {last_error}") from last_error


def embed_sentence_transformers(texts: list[str], model: str, batch_size: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install sentence-transformers to use the local backend; this backend is a documented deviation from EmoAgent."
        ) from exc
    encoder = SentenceTransformer(model)
    return np.asarray(
        encoder.encode(texts, batch_size=batch_size, normalize_embeddings=False, show_progress_bar=True),
        dtype=np.float32,
    )


def read_embedding_cache(path: Path, model: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    cache: dict[str, np.ndarray] = {}
    for record in read_jsonl(path):
        if record.get("model") != model:
            continue
        cache[record["node_id"]] = np.asarray(record["embedding"], dtype=np.float32)
    return cache


def append_embedding_cache(path: Path, node_id: str, model: str, embedding: np.ndarray) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "node_id": node_id,
                    "model": model,
                    "embedding": embedding.astype(float).tolist(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed EFK TextNodes for exact L2 retrieval")
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["volcengine-ark", "sentence-transformers"], default="volcengine-ark")
    parser.add_argument("--model", default=os.environ.get("ARK_EMBEDDING_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("ARK_BASE_URL"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-interval", type=float, default=0.05)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or ARK_EMBEDDING_MODEL is required")

    nodes = read_jsonl(args.nodes)
    if not nodes:
        raise ValueError("No TextNodes found; build and audit the knowledge base first")
    texts = [node["text"] for node in nodes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "embedding_cache.jsonl"

    if args.backend == "volcengine-ark":
        embedder = ArkTextEmbedder(args.model, args.base_url, args.max_retries)
        cached = read_embedding_cache(cache_path, args.model)
        for position, node in enumerate(nodes, 1):
            if node["node_id"] in cached:
                continue
            print(f"Ark embedding {position}/{len(nodes)}: {node['node_id']}", flush=True)
            vector = embedder.embed(node["text"])
            append_embedding_cache(cache_path, node["node_id"], args.model, vector)
            cached[node["node_id"]] = vector
            if args.request_interval > 0:
                time.sleep(args.request_interval)
        embeddings = np.stack([cached[node["node_id"]] for node in nodes])
    else:
        embeddings = embed_sentence_transformers(texts, args.model, args.batch_size)

    if args.normalize:
        embeddings = l2_normalize(embeddings)
    if len({len(vector) for vector in embeddings}) != 1:
        raise ValueError("Embedding dimensions are inconsistent")

    np.save(args.output_dir / "embeddings.npy", embeddings.astype(np.float32))
    write_json(args.output_dir / "node_ids.json", [node["node_id"] for node in nodes])
    write_json(
        args.output_dir / "embedding_manifest.json",
        {
            "backend": args.backend,
            "model": args.model,
            "base_url": args.base_url,
            "normalized": args.normalize,
            "node_count": len(nodes),
            "dimension": int(embeddings.shape[1]),
            "metric": "l2",
            "input_type": "text",
            "nodes_file": str(args.nodes.resolve()),
        },
    )
    print(json.dumps({"nodes": len(nodes), "shape": list(embeddings.shape)}, indent=2))


if __name__ == "__main__":
    main()
