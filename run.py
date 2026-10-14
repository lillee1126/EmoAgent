from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from emoagent.backends import ExactEFKRetriever, Flux2KleinEditor, QwenVLBackend, ResNetEmotionClassifier
from emoagent.constants import EMOTIONS
from emoagent.pipeline import EmoAgentPipeline, PipelineConfig


def parse_args() -> argparse.Namespace:
    base = Path(os.environ.get("EMOAGENT_BASE_DIR", Path(__file__).resolve().parent.parent))
    parser = argparse.ArgumentParser(description="Run the training-free EmoAgent reproduction on one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target-emotion", choices=EMOTIONS, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root", type=Path, default=base / "emoagent_runs")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--flux-model", default="black-forest-labs/FLUX.2-klein-9B")
    parser.add_argument("--qwen-device", default="cuda:0")
    parser.add_argument("--flux-device", default="cuda:1")
    parser.add_argument("--resnet-device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--flux-cpu-offload", action="store_true")
    parser.add_argument("--num-plans", type=int, default=5)
    parser.add_argument("--steps-per-plan", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-json-retries", type=int, default=3)
    parser.add_argument("--max-step-retries", type=int, default=2)
    parser.add_argument("--max-critic-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--require-resnet-target", action="store_true")
    parser.add_argument("--skip-resnet", action="store_true")
    parser.add_argument(
        "--efk-project-dir", type=Path, default=base / "emotion_factor_knowledge"
    )
    parser.add_argument(
        "--efk-nodes", type=Path, default=base / "efk_runs/main/knowledge/nodes.jsonl"
    )
    parser.add_argument("--efk-index", type=Path, default=base / "efk_runs/main/index")
    parser.add_argument(
        "--resnet-checkpoint",
        type=Path,
        default=base / "resnet50_emoset_classifier/runs/resnet50_emoset/best.pt",
    )
    parser.add_argument("--ark-base-url", default=os.environ.get("ARK_BASE_URL"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_plans < 1 or args.steps_per_plan < 1:
        raise ValueError("--num-plans and --steps-per-plan must be positive")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    for path in (args.efk_nodes, args.efk_index / "embeddings.npy"):
        if not path.exists():
            raise FileNotFoundError(f"Task-2 EFK artifact not found: {path}")
    if not args.skip_resnet and not args.resnet_checkpoint.is_file():
        raise FileNotFoundError(f"Task-1 checkpoint not found: {args.resnet_checkpoint}")

    # Load the VLM on GPU 0 and the editor on GPU 1, as requested by the task specification.
    vlm = QwenVLBackend(
        args.qwen_model,
        device=args.qwen_device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    editor = Flux2KleinEditor(
        args.flux_model,
        device=args.flux_device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        cpu_offload=args.flux_cpu_offload,
    )
    retriever = ExactEFKRetriever(
        args.efk_project_dir,
        args.efk_nodes,
        args.efk_index,
        args.ark_base_url,
    )
    classifier = None if args.skip_resnet else ResNetEmotionClassifier(
        args.resnet_checkpoint, args.resnet_device
    )
    config = PipelineConfig(
        output_root=args.output_root,
        num_plans=args.num_plans,
        steps_per_plan=args.steps_per_plan,
        top_k=args.top_k,
        max_json_retries=args.max_json_retries,
        max_step_retries=args.max_step_retries,
        max_critic_rounds=args.max_critic_rounds,
        seed=args.seed,
        require_resnet_target=args.require_resnet_target,
    )
    summary = EmoAgentPipeline(config, vlm, editor, retriever, classifier).run(
        args.image,
        args.target_emotion,
        args.run_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
