from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from efk.build_embeddings import extract_ark_embedding
from efk.common import write_json
from efk.qwen_summarize import extract_json, validate_summary
from efk.retrieve import ExactL2Retriever

try:
    import torch  # noqa: F401
except ImportError:
    greedy_centroid_clustering = None
else:
    from efk.clustering import greedy_centroid_clustering


class ClusteringTests(unittest.TestCase):
    @unittest.skipIf(greedy_centroid_clustering is None, "PyTorch is not installed locally")
    def test_greedy_centroid_threshold(self):
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.995, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        groups = greedy_centroid_clustering(vectors, threshold=0.95, block_size=2, device="cpu")
        self.assertEqual(sorted(sorted(group) for group in groups), [[0, 1], [2]])


class QwenValidationTests(unittest.TestCase):
    def test_markdown_json_is_accepted_and_normalized(self):
        raw = """```json
        {"cluster_summary":"People beneath gray skies.","factors":[
          {"element_type":"color tone","description":"muted gray color tones","evidence_count":4,"confidence":0.9}
        ]}
        ```"""
        value = validate_summary(extract_json(raw), image_count=5)
        self.assertEqual(value["factors"][0]["element_type"], "color_tone")

    def test_minority_factor_is_dropped_without_failing_cluster(self):
        value = {
            "cluster_summary": "A cluster summary.",
            "factors": [
                {
                    "element_type": "object",
                    "description": "one red flower",
                    "evidence_count": 2,
                    "confidence": 0.5,
                }
            ],
        }
        result = validate_summary(value, image_count=5)
        self.assertEqual(result["status"], "no_consensus")
        self.assertEqual(result["factors"], [])
        self.assertIn("evidence_count", result["dropped_factors"][0]["reason"])

    def test_concrete_one_word_and_same_type_factors_are_accepted(self):
        value = {
            "cluster_summary": "Fireworks and balloons fill the scene.",
            "factors": [
                {"element_type": "object", "description": "fireworks", "evidence_count": 4, "confidence": 0.9},
                {"element_type": "object", "description": "balloons", "evidence_count": 3, "confidence": 0.8},
            ],
        }
        result = validate_summary(value, image_count=5)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["factors"]), 2)


class RetrievalTests(unittest.TestCase):
    def test_metadata_filter_and_l2_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nodes = [
                {
                    "node_id": "sad_scene_1",
                    "text": "gray sky",
                    "emotion": "sadness",
                    "element_type": "scene",
                },
                {
                    "node_id": "sad_object_1",
                    "text": "wilted flowers",
                    "emotion": "sadness",
                    "element_type": "object",
                },
                {
                    "node_id": "awe_scene_1",
                    "text": "mountain summit",
                    "emotion": "awe",
                    "element_type": "scene",
                },
            ]
            nodes_path = root / "nodes.jsonl"
            nodes_path.write_text(
                "".join(json.dumps(node) + "\n" for node in nodes), encoding="utf-8"
            )
            np.save(root / "embeddings.npy", np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            write_json(root / "node_ids.json", [node["node_id"] for node in nodes])
            write_json(
                root / "embedding_manifest.json",
                {
                    "backend": "volcengine-ark",
                    "model": "test-ark-endpoint",
                    "normalized": False,
                },
            )
            retriever = ExactL2Retriever(nodes_path, root)
            result = retriever.search(
                np.asarray([0.1, 0.0], dtype=np.float32),
                emotion="sadness",
                element_type="scene",
                top_k=5,
            )
            self.assertEqual([item["node_id"] for item in result], ["sad_scene_1"])


class ArkResponseTests(unittest.TestCase):
    def test_documented_object_shape(self):
        vector = extract_ark_embedding({"data": {"embedding": [[0.25, -0.5, 1.0]]}})
        np.testing.assert_allclose(vector, [0.25, -0.5, 1.0])

    def test_list_data_shape(self):
        vector = extract_ark_embedding({"data": [{"embedding": [0.1, 0.2]}]})
        np.testing.assert_allclose(vector, [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
