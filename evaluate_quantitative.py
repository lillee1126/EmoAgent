"""Evaluate retained batch outputs; explicitly report engineering metric variants."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def emotion_metrics(source, edited, target):
    if set(source) != set(edited) or target not in source:
        raise ValueError("Emotion labels differ or target is missing")
    for distribution in (source, edited):
        if any(not math.isfinite(v) or v < 0 for v in distribution.values()):
            raise ValueError("Invalid emotion probabilities")
        if not math.isclose(sum(distribution.values()), 1, abs_tol=1e-4):
            raise ValueError("Emotion probabilities do not sum to one")
    def kl(p, q):
        return sum(v * math.log(v / max(q[k], 1e-12)) for k, v in p.items() if v > 0)
    return {
        "Emo-A_ResNet": float(max(edited, key=edited.get) == target),
        "Emo-S_ResNet_probability_delta": edited[target] - source[target],
        "ESR_ResNet_KL_edited_source": kl(edited, source),
        "KL_source_edited": kl(source, edited),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=20)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Load metric models and exit before inference")
    args = parser.parse_args()
    if not args.preflight_only:
        manifest = read(args.process_root / "batch_manifest.json")
        if manifest["selected_image_count"] != args.expected_images:
            raise ValueError("Manifest sample count differs from requested count")
        if manifest["total_pairs"] != args.expected_images:
            raise ValueError("Evaluation expects exactly one target emotion per source")
        if manifest["paper_workflow"]["plans_per_pair"] != 1:
            raise ValueError("This main-metric experiment requires exactly one plan per source")
        selections = sorted(args.process_root.rglob("selection.json"))
        if len(selections) != args.expected_images:
            raise ValueError(f"Only {len(selections)}/{args.expected_images} selections; finish inference first")

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    device = torch.device(args.device)
    clip = CLIPModel.from_pretrained(args.clip_model, local_files_only=args.local_files_only).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model, local_files_only=args.local_files_only)
    if args.preflight_only:
        print("CLIP model loaded successfully (single-plan main metrics)", flush=True)
        return

    def load_image(path):
        with Image.open(path) as im:
            return im.convert("RGB").copy()

    def features(paths):
        inputs = processor(images=[load_image(p) for p in paths], return_tensors="pt").to(device)
        with torch.inference_mode():
            value = clip.get_image_features(**inputs)
            return torch.nn.functional.normalize(value, dim=-1)

    rows = []
    for selection_path in selections:
        selection = read(selection_path)
        relative_pair = selection_path.parent.relative_to(args.process_root)
        attempt = selection_path.parent / selection["attempt"]
        summary = read(attempt / "summary.json")
        source = attempt / "source.png"
        result = args.result_root / relative_pair.parent / (relative_pair.name + ".png")
        variants = [attempt / item["plan_id"] / "final.png" for item in summary["results"]]
        expected_plans = manifest["paper_workflow"]["plans_per_pair"]
        if len(variants) != expected_plans:
            raise ValueError(f"Incomplete single-plan result: {relative_pair}")
        for path in [source, result, *variants]:
            if not path.is_file():
                raise FileNotFoundError(f"Missing metric image; use --retain-metric-images: {path}")
        source_probs = read(attempt / "source_emotion_probabilities.json")
        row = {"source_id": relative_pair.parent.as_posix(), "target_emotion": selection["target_emotion"],
               "selected_plan": selection["selected"]["plan_id"], "status": selection["selected"]["status"],
               "variant_count": len(variants), "result_path": str(result)}
        row.update(emotion_metrics(source_probs, selection["selected"]["final_probabilities"], selection["target_emotion"]))
        embedding = features([source, result])
        row["CLIP-I"] = float((embedding[0] * embedding[1]).sum().item())
        rows.append(row)
        print(f"Evaluated {len(rows)}/{len(selections)}: {relative_pair}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ["CLIP-I", "Emo-A_ResNet", "Emo-S_ResNet_probability_delta", "ESR_ResNet_KL_edited_source", "KL_source_edited"]
    report = {
        "images": len(rows), "seed": manifest["sampling_seed"],
        "means": {key: mean(row[key] for row in rows) for key in metrics},
        "per_emotion": {emotion: {"n": len(group), **{key: mean(row[key] for row in group) for key in metrics}}
                        for emotion in sorted({row["target_emotion"] for row in rows})
                        for group in [[row for row in rows if row["target_emotion"] == emotion]]},
        "clip_model": args.clip_model, "plans_per_image": 1,
        "not_applicable": {"LPIPS": "Single output per source; diversity experiment not performed.", "Sem-D": "Single output per source; diversity experiment not performed."},
        "definitions": {
            "Emo-A_ResNet": "Top-1 target accuracy, fraction in [0,1]; ResNet substitute, paper uses EmoVIT.",
            "Emo-S": "Engineering convention: p_edited(target) - p_source(target).",
            "ESR": "Engineering convention: KL(p_edited || p_source), natural log; reverse KL also saved because paper prose does not specify direction.",
            "CLIP-I": "Cosine similarity between original and selected output; CLIP checkpoint is an explicit engineering choice.",
            "selection": "One plan per source; original pipeline preservation gate may return the source image if the plan is rejected. Such fallback outputs remain in the denominator and status is retained in per_image.csv.",
            "scope": "20-source pilot; not directly comparable to the paper's full benchmark. Human study is not performed.",
        },
        "versions": {"torch": torch.__version__},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["means"], indent=2))


if __name__ == "__main__":
    main()
