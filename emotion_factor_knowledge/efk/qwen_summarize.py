from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


from .common import read_json, stable_id, write_json
from .constants import ELEMENT_TYPES, EMOTIONS


SYSTEM_PROMPT = """You build an evidence-grounded emotion-factor knowledge base.
The images in one request belong to the same EmoSet emotion and the same CLIP cluster.
Extract only concrete visual factors shared by a majority of the supplied images.
Do not use an abstract emotion word itself as a factor. Do not invent absent content.
Return JSON only, without Markdown."""


def build_prompt(emotion: str, image_count: int, correction: str | None = None) -> str:
    allowed = ", ".join(ELEMENT_TYPES)
    return f"""Target EmoSet emotion: {emotion}
Number of sampled images from this retained cluster: {image_count}

Summarize the cluster and extract consistent information across the five dimensions
specified by the EmoAgent appendix. A cluster may support more than one dimension,
may contain multiple distinct factors of the same element_type, and should omit
unsupported dimensions.

Allowed element_type values: {allowed}

Required schema:
{{
  "cluster_summary": "one concise, concrete English sentence",
  "factors": [
    {{
      "element_type": "one allowed value",
      "description": "a concise, directly editable English visual factor",
      "evidence_count": 3,
      "confidence": 0.0
    }}
  ]
}}

Rules:
- evidence_count must be an integer from 1 to {image_count}.
- Keep only factors visible in at least half of the supplied images (rounded up).
- confidence must be between 0 and 1.
- Do not mention filenames, CLIP, clustering, or this prompt.
- If no factor has majority evidence, return an empty factors list.
- Return valid JSON only.""" + (
        f"\nThe previous response failed validation: {correction}. Correct that error." if correction else ""
    )


def extract_json(text: str) -> dict[str, Any]:
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
        raise ValueError("response root must be an object")
    return value


def validate_summary(value: dict[str, Any], image_count: int) -> dict[str, Any]:
    summary = value.get("cluster_summary")
    factors = value.get("factors")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("cluster_summary must be a non-empty string")
    if not isinstance(factors, list):
        raise ValueError("factors must be a list")
    minimum_evidence = (image_count + 1) // 2
    normalized = []
    dropped = []
    seen_factors: set[tuple[str, str]] = set()
    for position, factor in enumerate(factors):
        if not isinstance(factor, dict):
            dropped.append({"position": position, "reason": "factor is not an object"})
            continue
        element_type = str(factor.get("element_type", "")).strip().lower().replace(" ", "_")
        if element_type not in ELEMENT_TYPES:
            dropped.append({"position": position, "reason": f"invalid element_type: {element_type!r}"})
            continue
        description = str(factor.get("description", "")).strip()
        if not description:
            dropped.append({"position": position, "reason": "factor description is empty"})
            continue
        key = (element_type, re.sub(r"\s+", " ", description).casefold())
        if key in seen_factors:
            dropped.append({"position": position, "reason": "exact duplicate factor"})
            continue
        try:
            evidence_count = int(factor.get("evidence_count"))
        except (TypeError, ValueError):
            dropped.append({"position": position, "reason": "evidence_count is not an integer"})
            continue
        if not minimum_evidence <= evidence_count <= image_count:
            dropped.append({
                "position": position,
                "reason": f"evidence_count must be in [{minimum_evidence}, {image_count}]",
            })
            continue
        try:
            confidence = float(factor.get("confidence"))
        except (TypeError, ValueError):
            dropped.append({"position": position, "reason": "confidence is not numeric"})
            continue
        if not 0.0 <= confidence <= 1.0:
            dropped.append({"position": position, "reason": "confidence must be between 0 and 1"})
            continue
        seen_factors.add(key)
        normalized.append(
            {
                "element_type": element_type,
                "description": description,
                "evidence_count": evidence_count,
                "confidence": confidence,
            }
        )
    return {
        "cluster_summary": summary.strip(),
        "status": "ok" if normalized else "no_consensus",
        "factors": normalized,
        "dropped_factors": dropped,
    }


class SummaryValidationError(ValueError):
    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class QwenClusterSummarizer:
    def __init__(
        self,
        model_name: str,
        device_map: str,
        dtype: str,
        min_pixels: int,
        max_pixels: int,
        max_new_tokens: int,
    ) -> None:
        from qwen_vl_utils import process_vision_info
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.process_vision_info = process_vision_info
        self.torch = torch
        self.max_new_tokens = max_new_tokens

    def summarize(
        self, paths: list[str], emotion: str, correction: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": build_prompt(emotion, len(paths), correction)}
        ]
        for path in paths:
            content.append({"type": "image", "image": Path(path).resolve().as_uri()})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        raw_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            validated = validate_summary(extract_json(raw_text), len(paths))
        except Exception as exc:
            raise SummaryValidationError(str(exc), raw_text) from exc
        return raw_text, validated


def sample_cluster_images(cluster: dict, count: int, seed: int) -> list[str]:
    paths = list(cluster["image_paths"])
    if len(paths) < count:
        raise ValueError(f"{cluster['cluster_id']} contains only {len(paths)} images")
    cluster_seed = int(stable_id(cluster["cluster_id"], str(seed)), 16)
    return random.Random(cluster_seed).sample(paths, count)


def main() -> None:
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Summarize every retained cluster with Qwen2.5-VL")
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--images-per-cluster", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--emotions", nargs="+", choices=EMOTIONS, default=list(EMOTIONS))
    args = parser.parse_args()

    summarizer = QwenClusterSummarizer(
        args.model,
        args.device_map,
        args.dtype,
        args.min_pixels,
        args.max_pixels,
        args.max_new_tokens,
    )
    totals = {}
    for emotion in args.emotions:
        source = read_json(args.filtered_dir / f"{emotion}.json")
        emotion_dir = args.output_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        for cluster in tqdm(source["clusters"], desc=f"Qwen {emotion}"):
            output_path = emotion_dir / f"{cluster['cluster_id']}.json"
            failed_path = output_path.with_suffix(".failed.json")
            if output_path.is_file():
                failed_path.unlink(missing_ok=True)
                continue
            sampled_paths = sample_cluster_images(cluster, args.images_per_cluster, args.seed)
            errors, raw_text = [], ""
            summary = None
            for _ in range(args.max_retries):
                try:
                    correction = errors[-1] if errors else None
                    raw_text, summary = summarizer.summarize(sampled_paths, emotion, correction)
                    break
                except Exception as exc:
                    if isinstance(exc, SummaryValidationError):
                        raw_text = exc.raw_response
                    errors.append(f"{type(exc).__name__}: {exc}")
            if summary is None:
                write_json(
                    failed_path,
                    {
                        "cluster_id": cluster["cluster_id"],
                        "emotion": emotion,
                        "sampled_image_paths": sampled_paths,
                        "raw_response": raw_text,
                        "errors": errors,
                    },
                )
                continue
            write_json(
                output_path,
                {
                    "cluster_id": cluster["cluster_id"],
                    "emotion": emotion,
                    "cluster_size": cluster["size"],
                    "cluster_cohesion": cluster["cohesion"],
                    "filter_metrics": cluster.get("filter_metrics", {}),
                    "sample_strategy": "random_without_replacement",
                    "sample_seed": args.seed,
                    "sampled_image_paths": sampled_paths,
                    "model": args.model,
                    **summary,
                },
            )
            failed_path.unlink(missing_ok=True)
        totals[emotion] = len(
            [path for path in emotion_dir.glob("*_cluster_*.json") if not path.name.endswith(".failed.json")]
        )
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
