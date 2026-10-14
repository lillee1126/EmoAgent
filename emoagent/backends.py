from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .constants import EMOTIONS
from .io_utils import extract_json_object


class VisionLanguageBackend(Protocol):
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]: ...


class ImageEditorBackend(Protocol):
    def edit(self, image_path: Path, prompt: str, output_path: Path, seed: int) -> None: ...


class QwenVLBackend:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 768 * 28 * 28,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Install task-3 dependencies before loading Qwen2.5-VL: pip install -r requirements.txt"
            ) from exc
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.torch = torch
        self.device = device
        self.process_vision_info = process_vision_info
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map={"": device},
            local_files_only=local_files_only,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=local_files_only,
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for path in image_paths or []:
            content.append({"type": "image", "image": path.resolve().as_uri()})
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        raw_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return raw_text, extract_json_object(raw_text)


class Flux2KleinEditor:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:1",
        dtype: str = "bfloat16",
        local_files_only: bool = False,
        cpu_offload: bool = False,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
    ) -> None:
        try:
            import torch
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise RuntimeError(
                "FLUX.2-klein requires a current Diffusers build. Install task-3 requirements first."
            ) from exc
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.torch = torch
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        if cpu_offload:
            gpu_id = int(device.split(":", 1)[1]) if device.startswith("cuda:") else 0
            self.pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            self.pipe.to(device)

    def edit(self, image_path: Path, prompt: str, output_path: Path, seed: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            source = image.convert("RGB")
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        with self.torch.inference_mode():
            result = self.pipe(
                image=source,
                prompt=prompt,
                height=source.height,
                width=source.width,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
            ).images[0]
        result.convert("RGB").save(output_path, format="PNG")


class ExactEFKRetriever:
    def __init__(self, project_dir: Path, nodes_path: Path, index_dir: Path, base_url: str | None) -> None:
        project = str(project_dir.resolve())
        if project not in sys.path:
            sys.path.insert(0, project)
        try:
            from efk.retrieve import ExactL2Retriever
        except ImportError as exc:
            raise RuntimeError(f"Cannot import task-2 EFK package from {project_dir}") from exc
        self.retriever = ExactL2Retriever(nodes_path, index_dir)
        self.base_url = base_url

    def retrieve_diverse(
        self,
        query: str,
        emotion: str,
        top_k: int = 5,
        allowed_element_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            return []
        query = " ".join(str(query).split())
        if not query:
            raise ValueError("neutral retrieval query is empty")
        allowed = list(allowed_element_types or ())
        if not allowed:
            allowed = sorted(
                {
                    str(node["element_type"])
                    for node in self.retriever.nodes
                    if node.get("emotion") == emotion
                }
            )

        # Appendix Eq. (4): encode source-image semantic text once. Emotion and
        # element type remain metadata constraints instead of query suffixes.
        vector = self.retriever.encode_query(query, self.base_url)
        per_type_depth = max(top_k * 3, 10)
        pool: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()
        for element_type in allowed:
            for item in self.retriever.search(
                vector, emotion, element_type, top_k=per_type_depth
            ):
                key = str(item.get("node_id") or item.get("text"))
                if key in seen_nodes:
                    continue
                seen_nodes.add(key)
                pool.append(item)

        def tokens(item: dict[str, Any]) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", str(item.get("text", "")).casefold()))

        def lexical_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
            left_tokens, right_tokens = tokens(left), tokens(right)
            if not left_tokens or not right_tokens:
                return 0.0
            return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

        # Greedy relevance/diversity selection over one combined candidate pool.
        # This is not one fixed top-1 per dimension: a dimension may contribute
        # multiple factors when they are relevant and not redundant.
        selected: list[dict[str, Any]] = []
        type_counts: dict[str, int] = {}
        remaining = list(pool)
        while remaining and len(selected) < top_k:
            def selection_score(item: dict[str, Any]) -> tuple[float, float, str]:
                element_type = str(item.get("element_type", ""))
                redundancy = max(
                    (lexical_similarity(item, chosen) for chosen in selected),
                    default=0.0,
                )
                item_clusters = set(map(str, item.get("source_cluster_ids") or []))
                cluster_overlap = max(
                    (
                        float(bool(item_clusters & set(map(str, chosen.get("source_cluster_ids") or []))))
                        for chosen in selected
                    ),
                    default=0.0,
                )
                type_repeat = type_counts.get(element_type, 0)
                coverage_bonus = -0.08 if type_repeat == 0 else 0.0
                score = (
                    float(item.get("l2_distance", 0.0))
                    + 0.30 * redundancy
                    + 0.12 * cluster_overlap
                    + 0.16 * type_repeat
                    + coverage_bonus
                )
                return score, float(item.get("l2_distance", 0.0)), str(item.get("node_id", ""))

            chosen = min(remaining, key=selection_score)
            remaining.remove(chosen)
            element_type = str(chosen.get("element_type", ""))
            type_counts[element_type] = type_counts.get(element_type, 0) + 1
            selected.append(
                {
                    **chosen,
                    "retrieval_query": query,
                    "selection_rank": len(selected) + 1,
                    "selection_policy": "global_exact_l2_then_diversity_rerank",
                }
            )
        return selected


class ResNetEmotionClassifier:
    def __init__(self, checkpoint_path: Path, device: str = "cuda:0") -> None:
        try:
            import torch
            from torch import nn
            from torchvision import transforms
            from torchvision.models import resnet50
        except ImportError as exc:
            raise RuntimeError("Install torch and torchvision to load the task-1 ResNet-50") from exc
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(EMOTIONS))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(224 * 256 / 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def predict(self, image_path: Path) -> dict[str, float]:
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0].detach().cpu().numpy()
        return {emotion: float(probabilities[index]) for index, emotion in enumerate(EMOTIONS)}


def empty_emotion_probabilities() -> dict[str, float]:
    return {emotion: float(value) for emotion, value in zip(EMOTIONS, np.zeros(len(EMOTIONS)))}
