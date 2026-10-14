from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

from emoagent.backends import ExactEFKRetriever, Flux2KleinEditor, QwenVLBackend, ResNetEmotionClassifier
from emoagent.constants import EMOTIONS
from emoagent.io_utils import append_jsonl, write_json
from emoagent.pipeline import EmoAgentPipeline, PipelineConfig
from emoagent.source_images import SourceImage, discover_source_images


BASE = Path(os.environ.get("EMOAGENT_BASE_DIR", Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-style EmoAgent workflow over PNG images and selected target emotions"
    )
    parser.add_argument("--input-dir", type=Path, default=BASE / "EmoEdit-inference/Original_405")
    parser.add_argument("--process-root", type=Path, default=BASE / "process")
    parser.add_argument("--result-root", type=Path, default=BASE / "result")
    parser.add_argument("--emotions", nargs="+", choices=EMOTIONS, default=list(EMOTIONS))
    parser.add_argument(
        "--random-one-emotion",
        action="store_true",
        help="Assign exactly one deterministic random target emotion to each input image.",
    )
    parser.add_argument("--num-plans", type=int, default=5)
    parser.add_argument("--steps-per-plan", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-json-retries", type=int, default=3)
    parser.add_argument("--max-step-retries", type=int, default=2)
    parser.add_argument("--max-critic-rounds", type=int, default=2)
    parser.add_argument("--pair-attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument(
        "--sample-images",
        type=int,
        default=0,
        help="Deterministically sample this many source images before target-emotion assignment.",
    )
    parser.add_argument("--require-image-count", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the batch after a pair exhausts --pair-attempts; the wrapper may then shut down the instance.",
    )
    parser.add_argument("--keep-intermediate-images", action="store_true")
    parser.add_argument(
        "--retain-metric-images",
        action="store_true",
        help="Keep source.png and every plan final.png, while compacting step-level PNGs.",
    )
    parser.add_argument("--shutdown-on-success", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--require-resnet-target", action="store_true")
    parser.add_argument("--skip-resnet", action="store_true")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--flux-model", default="black-forest-labs/FLUX.2-klein-9B")
    parser.add_argument("--qwen-device", default="cuda:0")
    parser.add_argument("--flux-device", default="cuda:1")
    parser.add_argument("--resnet-device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--flux-cpu-offload", action="store_true")
    parser.add_argument("--efk-project-dir", type=Path, default=BASE / "emotion_factor_knowledge")
    parser.add_argument("--efk-nodes", type=Path, default=BASE / "efk_runs/main/knowledge/nodes.jsonl")
    parser.add_argument("--efk-index", type=Path, default=BASE / "efk_runs/main/index")
    parser.add_argument(
        "--resnet-checkpoint",
        type=Path,
        default=BASE / "resnet50_emoset_classifier/runs/resnet50_emoset/best.pt",
    )
    parser.add_argument("--ark-base-url", default=os.environ.get("ARK_BASE_URL"))
    return parser.parse_args()


def numeric_image_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem.casefold())


def source_image_key(source: SourceImage) -> tuple[str, int, int | str]:
    parent = source.relative_path.parent.as_posix().casefold()
    numeric_group, stem = numeric_image_key(source.relative_path)
    return parent, numeric_group, stem


def sample_source_images(
    sources: list[SourceImage], count: int, seed: int
) -> list[SourceImage]:
    if count < 0:
        raise ValueError("sample count must be non-negative")
    if count > len(sources):
        raise ValueError(
            f"sample count {count} exceeds available images {len(sources)}"
        )
    if count == 0:
        return list(sources)
    return sorted(random.Random(seed).sample(sources, count), key=source_image_key)


def validate_args(args: argparse.Namespace, sources: list[SourceImage]) -> None:
    if not sources:
        raise FileNotFoundError(f"No PNG images found in {args.input_dir}")
    if args.require_image_count and len(sources) != args.require_image_count:
        raise ValueError(
            f"Expected exactly {args.require_image_count} PNG files, "
            f"found {len(sources)} in {args.input_dir}"
        )
    if min(
        args.num_plans,
        args.steps_per_plan,
        args.max_json_retries,
        args.pair_attempts,
    ) < 1:
        raise ValueError("plan, step, JSON-retry and pair-attempt counts must be positive")
    for path in (args.efk_nodes, args.efk_index / "embeddings.npy"):
        if not path.exists():
            raise FileNotFoundError(f"Task-2 EFK artifact not found: {path}")
    if not args.skip_resnet and not args.resnet_checkpoint.is_file():
        raise FileNotFoundError(f"Task-1 checkpoint not found: {args.resnet_checkpoint}")
    if not os.environ.get("ARK_API_KEY"):
        raise EnvironmentError("ARK_API_KEY is required for the task-2 EFK embedding query")


def next_attempt_number(pair_dir: Path) -> int:
    numbers: list[int] = []
    for path in pair_dir.glob("attempt_*"):
        try:
            numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=0) + 1


def select_final(summary: dict[str, Any], target_emotion: str) -> dict[str, Any]:
    candidates = summary.get("results")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Pipeline summary contains no candidate results")

    def score(candidate: dict[str, Any]) -> tuple[int, int, float]:
        probabilities = candidate.get("final_probabilities") or {}
        return (
            int(candidate.get("preservation_gate_passed", False)),
            int(candidate.get("status") == "accepted"),
            float(probabilities.get(target_emotion, 0.0)),
        )

    safe_candidates = [candidate for candidate in candidates if candidate.get("preservation_gate_passed")]
    if safe_candidates:
        selected = max(safe_candidates, key=score)
        fallback_used = False
    else:
        source_image = summary.get("source_image")
        if not source_image:
            raise ValueError("No candidate passed preservation and pipeline summary has no source fallback")
        selected = {
            "plan_id": "source_fallback",
            "status": "preservation_fallback",
            "final_image": str(source_image),
            "preservation_gate_passed": True,
            "final_probabilities": summary.get("source_probabilities") or {},
        }
        fallback_used = True
    selected_path = Path(str(selected["final_image"]))
    if not selected_path.is_file():
        raise FileNotFoundError(f"Selected candidate image is missing: {selected_path}")
    with Image.open(selected_path) as image:
        image.verify()
    return {
        "selection_rule": (
            "Require strict subject/entity/composition/content/brightness preservation first, then prefer "
            "Critic-accepted candidates, then maximize the task-1 ResNet target-emotion probability. "
            "The paper retains all diverse outputs and does not define a single-output selector; this rule is an "
            "engineering adaptation required by result/<image_id>/<emotion>.png."
        ),
        "source_fallback_used": fallback_used,
        "selected": selected,
        "candidate_scores": [
            {
                "plan_id": candidate.get("plan_id"),
                "status": candidate.get("status"),
                "preservation_gate_passed": bool(candidate.get("preservation_gate_passed", False)),
                "target_probability": float(
                    (candidate.get("final_probabilities") or {}).get(target_emotion, 0.0)
                ),
                "final_image": candidate.get("final_image"),
            }
            for candidate in candidates
        ],
    }


def compact_attempt_images(attempt_dir: Path, retain_metric_images: bool = False) -> dict[str, Any]:
    removed: list[str] = []
    retained: list[str] = []
    bytes_removed = 0
    for path in attempt_dir.rglob("*.png"):
        if not path.is_file():
            continue
        relative = path.relative_to(attempt_dir)
        keep_for_metrics = retain_metric_images and (
            relative == Path("source.png")
            or (path.name == "final.png" and any(part.startswith("plan_") for part in relative.parts))
        )
        if keep_for_metrics:
            retained.append(str(relative))
            continue
        bytes_removed += path.stat().st_size
        removed.append(str(relative))
        path.unlink()
    return {
        "policy": (
            "Keep source.png and each plan final.png for metrics; remove step-level PNGs"
            if retain_metric_images
            else "Keep JSON, JSONL and prompts; remove intermediate PNG copies after final selection"
        ),
        "removed_png_count": len(removed),
        "bytes_removed": bytes_removed,
        "removed_paths": removed,
        "retained_png_count": len(retained),
        "retained_paths": retained,
    }


def load_or_create_random_assignments(
    images: list[Path] | list[SourceImage], emotions: list[str], assignment_path: Path, seed: int
) -> dict[str, str]:
    expected_names = [
        source.assignment_key if isinstance(source, SourceImage) else source.name
        for source in images
    ]
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("Input images do not have unique persisted assignment keys")
    if assignment_path.is_file():
        payload = json.loads(assignment_path.read_text(encoding="utf-8"))
        assignments = payload.get("assignments")
        if not isinstance(assignments, dict):
            raise ValueError(f"Invalid assignment file: {assignment_path}")
        if set(assignments) != set(expected_names):
            raise ValueError(
                "Existing random-emotion assignment does not match the current input image set. "
                "Use a new process root instead of silently re-randomizing."
            )
        invalid = sorted({str(value) for value in assignments.values()} - set(emotions))
        if invalid:
            raise ValueError(f"Assignment file contains emotions outside --emotions: {invalid}")
        return {name: str(assignments[name]) for name in expected_names}

    rng = random.Random(seed)
    assignments = {name: rng.choice(emotions) for name in expected_names}
    counts = {emotion: sum(value == emotion for value in assignments.values()) for emotion in emotions}
    write_json(
        assignment_path,
        {
            "strategy": "independent_uniform_random_with_replacement",
            "seed": seed,
            "image_count": len(images),
            "candidate_emotions": emotions,
            "emotion_counts": counts,
            "assignments": assignments,
        },
    )
    return assignments


def release_transient_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_backends(args: argparse.Namespace):
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
    classifier = None
    if not args.skip_resnet:
        classifier = ResNetEmotionClassifier(args.resnet_checkpoint, args.resnet_device)
    return vlm, editor, retriever, classifier


def run_pair(
    args: argparse.Namespace,
    source: Path,
    image_index: int,
    emotion: str,
    emotion_index: int,
    backends,
    source_key: Path | None = None,
) -> tuple[bool, str]:
    source_key = source_key or Path(source.stem)
    pair_dir = args.process_root / source_key / emotion
    result_path = args.result_root / source_key / f"{emotion}.png"
    status_path = pair_dir / "status.json"
    pair_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.force and status_path.is_file() and result_path.is_file():
        try:
            prior = json.loads(status_path.read_text(encoding="utf-8"))
            completed_matches_request = (
                prior.get("state") == "completed"
                and Path(str(prior.get("source", ""))).resolve() == source.resolve()
                and prior.get("target_emotion") == emotion
            )
            if completed_matches_request:
                return True, "skipped_completed"
        except (OSError, json.JSONDecodeError):
            pass

    failures: list[dict[str, Any]] = []
    for _ in range(args.pair_attempts):
        attempt_number = next_attempt_number(pair_dir)
        attempt_name = f"attempt_{attempt_number:03d}"
        attempt_dir = pair_dir / attempt_name
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(
            status_path,
            {
                "state": "running",
                "source": str(source),
                "target_emotion": emotion,
                "active_attempt": attempt_name,
                "started_at": started_at,
                "prior_failures": failures,
            },
        )
        config = PipelineConfig(
            output_root=pair_dir,
            num_plans=args.num_plans,
            steps_per_plan=args.steps_per_plan,
            top_k=args.top_k,
            max_json_retries=args.max_json_retries,
            max_step_retries=args.max_step_retries,
            max_critic_rounds=args.max_critic_rounds,
            seed=args.seed + image_index * 100_000 + emotion_index * 1_000,
            require_resnet_target=args.require_resnet_target,
        )
        try:
            vlm, editor, retriever, classifier = backends
            summary = EmoAgentPipeline(config, vlm, editor, retriever, classifier).run(
                source,
                emotion,
                attempt_name,
            )
            selection = select_final(summary, emotion)
            selected_source = Path(str(selection["selected"]["final_image"]))
            shutil.copy2(selected_source, result_path)
            with Image.open(result_path) as image:
                image.verify()
            selection.update(
                {
                    "source": str(source),
                    "source_emotion": summary.get("source_emotion"),
                    "source_emotion_provenance": summary.get("source_emotion_provenance"),
                    "target_emotion": emotion,
                    "attempt": attempt_name,
                    "result_path": str(result_path),
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )
            write_json(pair_dir / "selection.json", selection)
            compact_record = None
            if not args.keep_intermediate_images:
                compact_record = compact_attempt_images(
                    attempt_dir,
                    retain_metric_images=args.retain_metric_images,
                )
                write_json(pair_dir / "compaction.json", compact_record)
            write_json(
                status_path,
                {
                    "state": "completed",
                    "source": str(source),
                    "source_emotion": summary.get("source_emotion"),
                    "source_emotion_provenance": summary.get("source_emotion_provenance"),
                    "target_emotion": emotion,
                    "successful_attempt": attempt_name,
                    "result_path": str(result_path),
                    "selected_plan": selection["selected"].get("plan_id"),
                    "selected_status": selection["selected"].get("status"),
                    "completed_at": selection["completed_at"],
                    "prior_failures": failures,
                    "compaction": compact_record,
                },
            )
            return True, "completed"
        except KeyboardInterrupt:
            write_json(
                status_path,
                {
                    "state": "interrupted",
                    "source": str(source),
                    "target_emotion": emotion,
                    "active_attempt": attempt_name,
                    "interrupted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            raise
        except Exception as exc:
            failure = {
                "attempt": attempt_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            if not args.keep_intermediate_images:
                failure["compaction"] = compact_attempt_images(
                    attempt_dir,
                    retain_metric_images=args.retain_metric_images,
                )
            failures.append(failure)
            write_json(attempt_dir / "error.json", failure)
            write_json(
                status_path,
                {
                    "state": "retrying",
                    "source": str(source),
                    "target_emotion": emotion,
                    "last_failure": failure,
                    "failures": failures,
                },
            )
            release_transient_memory()

    write_json(
        status_path,
        {
            "state": "failed",
            "source": str(source),
            "target_emotion": emotion,
            "failures": failures,
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return False, failures[-1]["error"] if failures else "unknown failure"


def shutdown_instance() -> None:
    subprocess.run(["sync"], check=False)
    shutdown = Path("/usr/bin/shutdown")
    if not shutdown.is_file():
        raise FileNotFoundError("AutoDL shutdown command not found: /usr/bin/shutdown")
    subprocess.run([str(shutdown)], check=True)


def main() -> int:
    args = parse_args()
    all_sources = sorted(
        discover_source_images(
            args.input_dir,
            excluded_roots=(args.process_root, args.result_root),
        ),
        key=source_image_key,
    )
    validate_args(args, all_sources)
    sources = sample_source_images(all_sources, args.sample_images, args.seed)
    if args.limit_images:
        sources = sources[: args.limit_images]
    args.process_root.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)
    events_path = args.process_root / "batch_events.jsonl"
    progress_path = args.process_root / "batch_progress.json"
    assignments: dict[str, str] | None = None
    if args.random_one_emotion:
        assignments = load_or_create_random_assignments(
            sources,
            list(args.emotions),
            args.process_root / "emotion_assignments.json",
            args.seed,
        )
        work_items = [
            (
                image_index,
                source,
                list(args.emotions).index(assignments[source.assignment_key]),
                assignments[source.assignment_key],
            )
            for image_index, source in enumerate(sources, 1)
        ]
    else:
        work_items = [
            (image_index, source, emotion_index, emotion)
            for image_index, source in enumerate(sources, 1)
            for emotion_index, emotion in enumerate(args.emotions)
        ]
    total = len(work_items)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_dir": str(args.input_dir),
        "source_emotion_mode": "planning_agent_vlm_prediction",
        "input_png_count": len(all_sources),
        "selected_image_count": len(sources),
        "sample_images": args.sample_images,
        "sampling_seed": args.seed,
        "sources": [
            {
                "path": str(source.path),
                "relative_path": source.assignment_key,
            }
            for source in sources
        ],
        "emotions": args.emotions,
        "total_pairs": total,
        "assignment_mode": "one_random_emotion_per_image" if args.random_one_emotion else "all_emotions_per_image",
        "assignment_file": str(args.process_root / "emotion_assignments.json") if assignments else None,
        "paper_workflow": {
            "pre_creation": "Planning Agent -> diverse plans -> Editing Agent -> independent Critic assessment",
            "optimization": "Each rejected candidate independently enters Editing/Critic iterative refinement",
            "plans_per_pair": args.num_plans,
            "steps_per_plan": args.steps_per_plan,
        },
        "task_adaptations": [
            "The paper's editing-tool library is replaced by task-specified FLUX.2-klein-9B.",
            "The paper retains all diverse final candidates; one result is selected per pair for the requested result tree.",
            "The unlabeled inference set uses the paper's Planning-Agent VLM source-emotion prediction.",
        ],
        "process_storage": (
            "keep all intermediate images"
            if args.keep_intermediate_images
            else (
                "keep source and all plan-final images for metrics; compact step-level PNGs"
                if args.retain_metric_images
                else "keep JSON/JSONL, compact intermediate PNGs"
            )
        ),
        "shutdown_on_success": args.shutdown_on_success,
    }
    write_json(args.process_root / "batch_manifest.json", manifest)

    print("Loading Qwen, FLUX, EFK retriever and ResNet once for the full batch...", flush=True)
    backends = load_backends(args)
    completed = skipped = failed = 0
    failures: list[dict[str, str]] = []

    for ordinal, (image_index, source_record, emotion_index, emotion) in enumerate(work_items, 1):
        source = source_record.path
        print(
            f"[{ordinal}/{total}] {source_record.assignment_key} -> {emotion}",
            flush=True,
        )
        started = time.time()
        ok, outcome = run_pair(
            args,
            source,
            image_index,
            emotion,
            emotion_index,
            backends,
            source_record.key,
        )
        if ok and outcome == "skipped_completed":
            skipped += 1
        elif ok:
            completed += 1
        else:
            failed += 1
            failures.append(
                {
                    "image": source_record.assignment_key,
                    "emotion": emotion,
                    "error": outcome,
                }
            )
        event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "ordinal": ordinal,
            "total": total,
            "image": source_record.assignment_key,
            "emotion": emotion,
            "ok": ok,
            "outcome": outcome,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        append_jsonl(events_path, event)
        write_json(
            progress_path,
            {
                "state": "running",
                "total": total,
                "processed": completed + skipped + failed,
                "completed_this_run": completed,
                "skipped_completed": skipped,
                "failed": failed,
                "last_event": event,
                "failures": failures,
            },
        )
        release_transient_memory()
        if not ok and args.fail_fast:
            raise RuntimeError(
                f"Fail-fast: {source_record.assignment_key} -> {emotion} exhausted "
                f"{args.pair_attempts} pair attempts: {outcome}"
            )

    success = failed == 0 and completed + skipped == total
    final_progress = {
        "state": "completed" if success else "completed_with_failures",
        "total": total,
        "completed_this_run": completed,
        "skipped_completed": skipped,
        "failed": failed,
        "failures": failures,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(progress_path, final_progress)
    print(json.dumps(final_progress, ensure_ascii=False, indent=2), flush=True)

    if success and args.shutdown_on_success:
        if args.require_image_count and len(all_sources) != args.require_image_count:
            raise RuntimeError("Shutdown refused because the required input-image count was not met")
        if (args.limit_images or args.sample_images) or set(args.emotions) != set(EMOTIONS):
            raise RuntimeError(
                "Shutdown refused because sampling/limit or a partial emotion list was used"
            )
        print("All requested pairs completed successfully. Shutting down the AutoDL instance...", flush=True)
        shutdown_instance()
    elif args.shutdown_on_success:
        print("Shutdown skipped because one or more pairs failed.", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
