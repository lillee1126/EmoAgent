from __future__ import annotations

import json
import re
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any

from PIL import Image

from batch_run import (
    compact_attempt_images,
    load_or_create_random_assignments,
    run_pair,
    sample_source_images,
    select_final,
)
from emoagent.io_utils import extract_json_object
from emoagent.pipeline import EmoAgentPipeline, PipelineConfig
from emoagent.source_images import SourceImage, discover_source_images
from emoagent.prompts import (
    PAPER_SOURCE_ANALYSIS,
    PAPER_SOURCE_QUERY,
    RUNTIME_SOURCE_QUERY,
    flux_prompt_generation,
    source_analysis_prompt,
)


def analysis_payload(*, malicious_source_emotion: str | None = "anger") -> dict[str, Any]:
    value: dict[str, Any] = {
        "main_subject": "one blue hot-air balloon",
        "objects": ["one hot-air balloon", "one park path", "trees", "distant vehicles"],
        "scene": "an outdoor park beneath a clear daytime sky",
        "composition": (
            "a low wide viewpoint with the balloon centered above a path and distant vehicles "
            "in the background"
        ),
        "object_attributes": ["blue-and-white balloon envelope", "curved paved path"],
        "expressions_actions": ["distant people stand near the path"],
        "visible_people_or_faces": [],
        "facial_expression_available": False,
        "protected_entities": [
            {
                "entity": "hot-air balloon",
                "approximate_count": "one",
                "preserve_identity": ["blue-and-white striped balloon envelope"],
                "preserve_geometry": ["centered above the path at the same scale"],
                "immutable_attributes": ["single balloon identity"],
                "editable_attributes": ["minor decorative accents"],
            },
            {
                "entity": "park path",
                "approximate_count": "one",
                "preserve_identity": ["paved path"],
                "preserve_geometry": ["curves from foreground toward the balloon"],
                "immutable_attributes": ["perspective direction"],
                "editable_attributes": [],
            },
        ],
        "must_preserve_layout": [
            "keep the balloon centered above the curved path",
            "keep trees and distant vehicles in their depth layers",
        ],
        "editable_or_replaceable_elements": [
            "open background area to the right of the balloon may receive one small secondary object",
            "sky lighting may be adjusted without changing geometry",
        ],
        "key_information_rationale": "the balloon and path perspective define the image identity",
        "lighting": "bright daylight with normal exposure",
        "global_hue": "blue sky, green trees, and a blue-and-white balloon",
        "semantic_summary": (
            "One blue-and-white hot-air balloon is centered above a curved park path with trees "
            "and distant vehicles in daylight."
        ),
        "retrieval_semantics": (
            "One blue-and-white hot-air balloon above a curved public-park path, with trees, "
            "distant vehicles, open background space to its right, and clear daylight."
        ),
    }
    if malicious_source_emotion is not None:
        value["source_emotion"] = malicious_source_emotion
        value["source_emotion_evidence"] = [
            "visible subject posture, lighting, and scene cues support the prediction"
        ]
    return value


def instruction_payload(
    *,
    step_id: int = 1,
    element_type: str = "object",
    method: str = "add",
    operation_target: str = "carousel",
    target_region: str = "the distant background to the right of the hot-air balloon",
    instruction: str = "Add one small carousel in the background to the right of the hot-air balloon.",
    target_emotion: str = "amusement",
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "element_type": element_type,
        "method": method,
        "operation_target": operation_target,
        "target_region": target_region,
        "instruction": instruction,
        "emotion_rationale": f"The visible operation supplies a concrete cue for {target_emotion}.",
        "preserve": [
            "hot-air balloon identity, position, scale, and silhouette",
            "park-path perspective",
            "trees and distant vehicles",
        ],
    }


def plan_payload(target_emotion: str = "amusement") -> dict[str, Any]:
    return {
        "plan_id": "model_chosen_id",
        "strategy_summary": "a small fairground cue behind the protected balloon",
        "strategy_anchor": "one secondary carousel in the right background",
        "distinctive_cues": ["right-background carousel"],
        "instructions": [instruction_payload(target_emotion=target_emotion)],
    }


def verification_spec() -> dict[str, Any]:
    return {
        "required_change": (
            "Exactly one recognizable carousel is present as a secondary object in the distant "
            "background to the right of the hot-air balloon."
        ),
        "target_region": "the open distant background to the right of the hot-air balloon",
        "placement_scale_depth": [
            "the carousel is smaller than the balloon and remains behind it",
            "the carousel does not overlap the balloon silhouette or park path",
        ],
        "must_remain_unchanged": [
            "the balloon retains its identity, count, position, scale, and silhouette",
            "the park-path perspective, trees, people, vehicles, lighting, and all other regions remain unchanged",
        ],
        "must_not_appear": [
            "no second carousel, extra large ride, balloon replacement, obstruction, or global redraw appears"
        ],
        "success_criteria": [
            "one recognizable carousel is visible in the specified right-background region",
            "the balloon and park-path composition remain unchanged and unobstructed",
        ],
    }


class ScriptedVLM:
    """A strict full-pipeline double that speaks only the current schemas."""

    def __init__(
        self,
        *,
        already_satisfied: bool = False,
        malformed_joint_evaluation: bool = False,
        malicious_source_emotion: str = "anger",
    ) -> None:
        self.already_satisfied = already_satisfied
        self.malformed_joint_evaluation = malformed_joint_evaluation
        self.malicious_source_emotion = malicious_source_emotion
        self.calls: list[dict[str, Any]] = []
        self.spec = verification_spec()

    @staticmethod
    def _target_from_prompt(user_prompt: str) -> str:
        for pattern in (
            r"User text prompt: make it ([a-z_]+)",
            r"Target emotion: ([a-z_]+)",
        ):
            match = re.search(pattern, user_prompt, re.IGNORECASE)
            if match:
                return match.group(1).casefold()
        return "amusement"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_paths": list(image_paths or []),
                "max_new_tokens": max_new_tokens,
            }
        )
        target_emotion = self._target_from_prompt(user_prompt)

        if "source-image analyst" in system_prompt:
            value = analysis_payload(malicious_source_emotion=self.malicious_source_emotion)
        elif "Translate a target emotion" in system_prompt:
            value = {
                "target_emotion": target_emotion,
                "plans": [plan_payload(target_emotion)],
            }
        elif "Planning_evaluate" in system_prompt:
            if self.malformed_joint_evaluation:
                value = {
                    "verdict": "revise",
                    "issues": ["intentionally incomplete evaluator response"],
                    "revised_plans": [],
                }
            else:
                value = {
                    "verdict": "reasonable",
                    "issues": [],
                    "revised_plans": [plan_payload(target_emotion)],
                }
        elif "Build instruction-specific verification criteria" in system_prompt:
            value = {
                "execution_decision": (
                    "already_satisfied" if self.already_satisfied else "edit"
                ),
                "decision_reason": (
                    "One carousel is already clearly visible in the exact right-background region."
                    if self.already_satisfied
                    else "No carousel is visible in the requested right-background region."
                ),
                "edit_prompt": (
                    ""
                    if self.already_satisfied
                    else (
                        "CONTROLLED REGION EDIT: Add exactly one small carousel to the distant "
                        "background right of the hot-air balloon. Keep it behind and smaller than "
                        "the balloon. Preserve the balloon, path, trees, people, vehicles, lighting, "
                        "viewpoint, scale, silhouettes, and every other region exactly."
                    )
                ),
                "verification_spec": self.spec,
            }
        elif "step-level self-critic" in system_prompt:
            criteria = self.spec["success_criteria"]
            value = {
                "verdict": "correct",
                "required_state_satisfied": True,
                "instruction_executed": not self.already_satisfied,
                "change_localized": True,
                "preservation_ok": True,
                "emotion_direction_ok": True,
                "facial_expression_constraint_obeyed": True,
                "main_subject_preserved": True,
                "key_information_preserved": True,
                "composition_preserved": True,
                "new_content_context_compatible": True,
                "brightness_and_detail_preserved": True,
                "criterion_results": [
                    {
                        "criterion": criterion,
                        "passed": True,
                        "evidence": f"Visible evidence satisfies: {criterion}",
                    }
                    for criterion in criteria
                ],
                "problems": [],
                "retry_instruction": "",
            }
        elif "Critic Agent in EmoAgent" in system_prompt:
            value = {
                "verdict": "correct",
                "perceived_emotion": target_emotion,
                "emotion_evidence": {
                    "objects": ["secondary carousel"],
                    "object_attributes": ["small scale"],
                    "expressions_actions": [],
                    "background": ["carousel behind balloon"],
                    "global_hue_lighting": ["source daylight retained"],
                },
                "semantic_preservation": "good",
                "facial_expression_constraint_obeyed": True,
                "main_subject_preserved": True,
                "key_information_preserved": True,
                "composition_preserved": True,
                "new_content_context_compatible": True,
                "multiple_emotion_factors_present": True,
                "brightness_and_detail_preserved": True,
                "instruction_diagnosis": [],
                "updated_instructions": [],
                "final_reason": "The requested cue is visible and protected content remains intact.",
            }
        else:
            raise AssertionError(f"Unexpected system prompt: {system_prompt}")
        return json.dumps(value), value


class FalseNoopVLM(ScriptedVLM):
    """Always claims no-op; the visual validator disproves only the first claim."""

    def __init__(self) -> None:
        super().__init__(already_satisfied=True)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        if "step-level self-critic" not in system_prompt:
            return super().generate_json(
                system_prompt, user_prompt, image_paths=image_paths, max_new_tokens=max_new_tokens
            )
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_paths": list(image_paths or []),
                "max_new_tokens": max_new_tokens,
            }
        )
        is_edit = "Execution decision: edit" in user_prompt
        criteria = self.spec["success_criteria"]
        value = {
            "verdict": "correct" if is_edit else "wrong",
            "required_state_satisfied": is_edit,
            "instruction_executed": is_edit,
            "change_localized": is_edit,
            "preservation_ok": True,
            "emotion_direction_ok": True,
            "facial_expression_constraint_obeyed": True,
            "main_subject_preserved": True,
            "key_information_preserved": True,
            "composition_preserved": True,
            "new_content_context_compatible": True,
            "brightness_and_detail_preserved": True,
            "criterion_results": [
                {
                    "criterion": criterion,
                    "passed": is_edit if index == 0 else True,
                    "evidence": (
                        "The requested carousel is visible after editing."
                        if is_edit
                        else "The requested carousel is not present before editing."
                    ),
                }
                for index, criterion in enumerate(criteria)
            ],
            "problems": [] if is_edit else ["The requested state is not already satisfied."],
            "retry_instruction": "Add the requested carousel." if not is_edit else "",
        }
        return json.dumps(value), value


class BrokenComposerVLM(ScriptedVLM):
    """Returns an invalid composer schema so deterministic fallback is exercised."""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        if "Build instruction-specific verification criteria" in system_prompt:
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "image_paths": list(image_paths or []),
                    "max_new_tokens": max_new_tokens,
                }
            )
            return "{}", {}
        return super().generate_json(
            system_prompt, user_prompt, image_paths=image_paths, max_new_tokens=max_new_tokens
        )


class BrokenStepValidatorVLM(ScriptedVLM):
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        if "step-level self-critic" in system_prompt:
            self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
            return "{}", {}
        return super().generate_json(
            system_prompt, user_prompt, image_paths=image_paths, max_new_tokens=max_new_tokens
        )


class BrokenFinalCriticVLM(ScriptedVLM):
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        if "Critic Agent in EmoAgent" in system_prompt:
            self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
            return "{}", {}
        return super().generate_json(
            system_prompt, user_prompt, image_paths=image_paths, max_new_tokens=max_new_tokens
        )


class RecordingEditor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def edit(self, image_path: Path, prompt: str, output_path: Path, seed: int) -> None:
        self.calls.append(
            {
                "image_path": image_path,
                "prompt": prompt,
                "output_path": output_path,
                "seed": seed,
            }
        )
        with Image.open(image_path) as image:
            image.save(output_path)


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve_diverse(
        self,
        query: str,
        emotion: str,
        top_k: int = 5,
        allowed_element_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "query": query,
                "emotion": emotion,
                "top_k": top_k,
                "allowed_element_types": list(allowed_element_types or []),
            }
        )
        return [
            {
                "node_id": "amusement_object_001",
                "text": "a small carousel",
                "element_type": "object",
                "emotion": emotion,
                "l2_distance": 0.1,
            }
        ]


class CoreTests(unittest.TestCase):
    def _run_pipeline(
        self,
        root: Path,
        *,
        vlm: ScriptedVLM | None = None,
        editor: RecordingEditor | None = None,
        retriever: RecordingRetriever | None = None,
        run_name: str = "test_run",
        target_emotion: str = "amusement",
        max_json_retries: int = 2,
        max_step_retries: int = 0,
    ) -> tuple[dict[str, Any], ScriptedVLM, RecordingEditor, RecordingRetriever]:
        source = root / "source.jpg"
        Image.new("RGB", (640, 480), (90, 150, 210)).save(source)
        vlm = vlm or ScriptedVLM()
        editor = editor or RecordingEditor()
        retriever = retriever or RecordingRetriever()
        config = PipelineConfig(
            output_root=root / "runs",
            num_plans=1,
            steps_per_plan=1,
            top_k=5,
            max_json_retries=max_json_retries,
            max_step_retries=max_step_retries,
            max_critic_rounds=0,
        )
        summary = EmoAgentPipeline(
            config,
            vlm,
            editor,
            retriever,
            emotion_classifier=None,
        ).run(
            source,
            target_emotion,
            run_name,
        )
        return summary, vlm, editor, retriever

    def test_json_extraction(self):
        self.assertEqual(extract_json_object('prefix ```json\n{"ok": true}\n```')["ok"], True)

    def test_runtime_analysis_predicts_source_emotion_without_target_bias(self):
        system, user = source_analysis_prompt()
        combined = f"{system}\n{user}"

        self.assertIn("What are the semantics", PAPER_SOURCE_QUERY)
        self.assertIn("color tone", PAPER_SOURCE_ANALYSIS)
        self.assertIn(RUNTIME_SOURCE_QUERY, user)
        self.assertIn("classify the source emotion", system)
        self.assertIn('"source_emotion"', combined)
        self.assertIn("mixed/uncertain", combined)
        self.assertNotIn("The later editing target is", combined)

    def test_vlm_source_emotion_is_recorded_with_prediction_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, _, _, _ = self._run_pipeline(
                root,
                vlm=ScriptedVLM(malicious_source_emotion="anger"),
            )

            analysis = json.loads(
                (root / "runs/test_run/source_analysis.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (root / "runs/test_run/run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["source_emotion"], "anger")
            self.assertEqual(analysis["source_emotion"], "anger")
            self.assertEqual(
                analysis["source_emotion_provenance"], "planning_agent_vlm_prediction"
            )
            self.assertEqual(manifest["source_emotion"], "anger")

    def test_source_analysis_tolerates_missing_evidence_and_affective_summary(self):
        value = analysis_payload(malicious_source_emotion="contentment")
        value.pop("source_emotion_evidence")
        value["semantic_summary"] = (
            "A peaceful boat is centered on calm water beneath a pale evening sky."
        )

        validated = EmoAgentPipeline._validate_analysis(value)

        self.assertTrue(validated["source_emotion_evidence"])
        self.assertIn("peaceful", validated["semantic_summary"].casefold())
        self.assertNotIn("peaceful", validated["retrieval_semantics"].casefold())

    def test_affective_retrieval_semantics_is_neutralized_instead_of_rejecting_image(self):
        value = analysis_payload(malicious_source_emotion="excitement")
        value["semantic_summary"] = "A vibrant balloon is centered above a park path."
        value["retrieval_semantics"] = "A vibrant exciting balloon above a park path."

        validated = EmoAgentPipeline._validate_analysis(value)

        self.assertTrue(validated["retrieval_semantics_normalized"])
        self.assertNotIn("vibrant", validated["retrieval_semantics"].casefold())
        self.assertNotIn("exciting", validated["retrieval_semantics"].casefold())

    def test_retrieval_uses_one_neutral_semantic_query_and_metadata_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retriever = RecordingRetriever()
            self._run_pipeline(root, retriever=retriever)

            self.assertEqual(len(retriever.calls), 1)
            call = retriever.calls[0]
            expected = " ".join(analysis_payload()["retrieval_semantics"].split())
            self.assertEqual(call["query"], expected)
            self.assertEqual(call["emotion"], "amusement")
            self.assertEqual(call["top_k"], 5)
            self.assertNotIn("amusement", call["query"].casefold())
            self.assertNotIn("contentment", call["query"].casefold())
            self.assertNotIn("describe how to modify", call["query"].casefold())
            self.assertNotIn("focusing on", call["query"].casefold())
            self.assertNotIn("facial_expression", call["allowed_element_types"])

            record = json.loads(
                (root / "runs/test_run/efk_retrieval.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["embedding_query"], expected)
            self.assertEqual(record["target_emotion"], "amusement")

    def test_dynamic_verification_spec_is_required_and_generic_placeholder_is_rejected(self):
        instruction = instruction_payload()
        valid_value = {
            "execution_decision": "edit",
            "decision_reason": "No carousel is visible in the requested region.",
            "edit_prompt": "CONTROLLED REGION EDIT: Add one small carousel to the right background.",
            "verification_spec": verification_spec(),
        }

        validated = EmoAgentPipeline._validate_flux_prompt(valid_value, instruction)
        self.assertIn("carousel", validated["verification_spec"]["required_change"].casefold())
        self.assertGreaterEqual(len(validated["verification_spec"]["success_criteria"]), 2)
        self.assertNotIn("expected_visible_change", validated)

        generic_value = {
            **valid_value,
            "verification_spec": {
                **verification_spec(),
                "required_change": "one objectively checkable visible result",
            },
        }
        with self.assertRaisesRegex(ValueError, "instruction-specific"):
            EmoAgentPipeline._validate_flux_prompt(generic_value, instruction)

    def test_already_satisfied_skips_editor_but_still_runs_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vlm = ScriptedVLM(already_satisfied=True)
            editor = RecordingEditor()
            summary, _, _, _ = self._run_pipeline(root, vlm=vlm, editor=editor)

            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(editor.calls, [])
            step_result = json.loads(
                (
                    root
                    / "runs/test_run/plan_01/pre_creation/step_01/step_result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(step_result["execution_decision"], "already_satisfied")
            self.assertTrue(step_result["accepted"])
            self.assertEqual(step_result["attempts"][0]["prompt"]["edit_prompt"], "")
            trajectory = (root / "runs/test_run/trajectory.jsonl").read_text(encoding="utf-8")
            self.assertIn("skip_edit_already_satisfied", trajectory)
            self.assertIn("edit_reactify", trajectory)

    def test_visually_disproved_noop_is_deterministically_converted_to_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vlm = FalseNoopVLM()
            editor = RecordingEditor()
            summary, _, _, _ = self._run_pipeline(
                root,
                vlm=vlm,
                editor=editor,
                max_step_retries=1,
            )

            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(len(editor.calls), 1)
            step_result = json.loads(
                (
                    root
                    / "runs/test_run/plan_01/pre_creation/step_01/step_result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(step_result["attempts"][0]["prompt"]["execution_decision"], "already_satisfied")
            forced = step_result["attempts"][1]["prompt"]
            self.assertEqual(forced["execution_decision"], "edit")
            self.assertTrue(forced["edit_prompt"].startswith("CONTROLLED REGION EDIT"))
            self.assertEqual(forced["decision_override"]["original_decision"], "already_satisfied")

    def test_broken_edit_composer_uses_deterministic_prompt_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            editor = RecordingEditor()
            summary, _, _, _ = self._run_pipeline(
                root,
                vlm=BrokenComposerVLM(),
                editor=editor,
                max_json_retries=2,
            )

            self.assertEqual(summary["accepted"], 1)
            self.assertGreaterEqual(len(editor.calls), 1)
            step_result = json.loads(
                (
                    root
                    / "runs/test_run/plan_01/pre_creation/step_01/step_result.json"
                ).read_text(encoding="utf-8")
            )
            fallback = step_result["attempts"][0]["prompt"]
            self.assertEqual(fallback["execution_decision"], "edit")
            self.assertIn("deterministic fallback", fallback["decision_override"]["policy"])
            trajectory = (root / "runs/test_run/trajectory.jsonl").read_text(encoding="utf-8")
            self.assertIn("compose_edit_prompt_fallback", trajectory)

    def test_broken_step_validator_rejects_and_rolls_back_without_pipeline_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, _, _, _ = self._run_pipeline(
                root,
                vlm=BrokenStepValidatorVLM(),
                max_json_retries=2,
            )

            self.assertEqual(summary["total"], 1)
            step_result = json.loads(
                (
                    root
                    / "runs/test_run/plan_01/pre_creation/step_01/step_result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertFalse(step_result["accepted"])
            self.assertTrue(step_result["rollback_applied"])
            self.assertIn(
                "validation_normalization",
                step_result["attempts"][0]["validation"],
            )

    def test_broken_final_critic_rejects_candidate_without_pipeline_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, _, _, _ = self._run_pipeline(
                root,
                vlm=BrokenFinalCriticVLM(),
                max_json_retries=2,
            )

            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["results"][0]["status"], "unresolved")
            feedback = json.loads(
                (
                    root
                    / "runs/test_run/plan_01/critic_feedback_round_00.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(feedback["verdict"], "wrong")
            self.assertIn("critic_normalization", feedback)

    def test_order_insensitive_duplicate_plans_are_reported_without_aborting_batch(self):
        root = Path("unused")
        pipeline = EmoAgentPipeline(
            PipelineConfig(output_root=root, num_plans=2, steps_per_plan=2),
            ScriptedVLM(),
            RecordingEditor(),
            RecordingRetriever(),
            emotion_classifier=None,
        )
        carousel = instruction_payload(step_id=1)
        lighting = instruction_payload(
            step_id=2,
            element_type="color_tone",
            method="filter_change",
            operation_target="scene lighting",
            target_region="the full frame",
            instruction="Shift the existing scene lighting toward a warm daylight treatment.",
        )
        plans = [
            {
                "strategy_summary": "carousel then lighting",
                "strategy_anchor": "fairground object path",
                "distinctive_cues": ["cue alpha", "cue beta"],
                "instructions": [carousel, lighting],
            },
            {
                "strategy_summary": "lighting then carousel",
                "strategy_anchor": "nominally different path",
                "distinctive_cues": ["cue gamma", "cue delta"],
                "instructions": [lighting, carousel],
            },
        ]

        validated = pipeline._validate_plans(
            {"target_emotion": "amusement", "plans": plans},
            analysis_payload(),
            "amusement",
        )
        warnings = validated["diversity_report"]["warnings"]
        self.assertTrue(any("same order-insensitive" in warning for warning in warnings))
        self.assertTrue(any("complete semantic duplicates" in warning for warning in warnings))

    def test_three_of_four_shared_operations_are_reported_not_rejected(self):
        pipeline = EmoAgentPipeline(
            PipelineConfig(output_root=Path("unused"), num_plans=2, steps_per_plan=4),
            ScriptedVLM(),
            RecordingEditor(),
            RecordingRetriever(),
            emotion_classifier=None,
        )

        def add_cue(name: str, step_id: int) -> dict[str, Any]:
            operations = {
                "carousel": "Place one compact carousel beyond the path without covering any subject.",
                "pinwheel": "Introduce several colorful pinwheels beside the existing walkway.",
                "bunting": "Suspend a short line of triangular bunting between two background trees.",
                "soap bubbles": "Add translucent soap bubbles drifting through the open midground air.",
                "kite": "Include one diamond-shaped kite high in the unobstructed sky.",
            }
            return instruction_payload(
                step_id=step_id,
                operation_target=name,
                target_region=f"a compatible open region for the {name}",
                instruction=operations[name],
            )

        shared = [add_cue("carousel", 1), add_cue("pinwheel", 2), add_cue("bunting", 3)]
        plans = [
            {
                "strategy_summary": "fairground details with floating bubbles",
                "strategy_anchor": "soap bubbles around small fairground details",
                "distinctive_cues": ["soap bubbles"],
                "instructions": [*shared, add_cue("soap bubbles", 4)],
            },
            {
                "strategy_summary": "fairground details with one small kite",
                "strategy_anchor": "one kite above small fairground details",
                "distinctive_cues": ["single kite"],
                "instructions": [*shared, add_cue("kite", 4)],
            },
        ]

        validated = pipeline._validate_plans(
            {"target_emotion": "amusement", "plans": plans},
            analysis_payload(),
            "amusement",
        )

        pair = validated["diversity_report"]["pairwise"][0]
        self.assertEqual(pair["shared_operations"], 3)
        self.assertTrue(pair["warning"])
        self.assertEqual(len(validated["plans"]), 2)

    def test_unlisted_protected_entity_attribute_uses_visual_review_not_hard_failure(self):
        pipeline = EmoAgentPipeline(
            PipelineConfig(output_root=Path("unused"), num_plans=1, steps_per_plan=1),
            ScriptedVLM(),
            RecordingEditor(),
            RecordingRetriever(),
            emotion_classifier=None,
        )
        analysis = analysis_payload()
        analysis["protected_entities"].append(
            {
                "entity": "newspaper",
                "approximate_count": "one",
                "preserve_identity": ["the same folded newspaper"],
                "preserve_geometry": ["same position, scale, and fold"],
                "immutable_attributes": ["newspaper identity and folded silhouette"],
                "editable_attributes": [],
            }
        )
        plan = {
            "strategy_summary": "localized playful printed detail",
            "strategy_anchor": "comic-style artwork on the existing newspaper",
            "distinctive_cues": ["newspaper comic panels"],
            "instructions": [
                instruction_payload(
                    element_type="object_attribute",
                    method="attribute_change",
                    operation_target="newspaper",
                    target_region="only the visible printed face of the existing newspaper",
                    instruction=(
                        "Change only the visible newspaper print into whimsical illustrated comic "
                        "panels while retaining the same newspaper, fold, position, and scale."
                    ),
                )
            ],
        }

        validated = pipeline._validate_plans(
            {"target_emotion": "amusement", "plans": [plan]}, analysis, "amusement"
        )

        review = validated["plans"][0]["instructions"][0]["constraint_review"]
        self.assertEqual(review["status"], "visual_validation_required")
        self.assertIn("non-exhaustive", review["warnings"][0])

    def test_taxonomy_mismatch_addition_labeled_background_change_is_rejected(self):
        mismatched = instruction_payload(
            element_type="background",
            method="background_change",
            operation_target="carousel",
            instruction="Add a carousel in the background to the right of the balloon.",
        )
        with self.assertRaisesRegex(ValueError, "addition/count increase"):
            EmoAgentPipeline._validate_instruction_taxonomy(mismatched)

    def test_plan_validation_repairs_addition_labeled_background_change(self):
        pipeline = EmoAgentPipeline(
            PipelineConfig(output_root=Path("unused"), num_plans=1, steps_per_plan=1),
            ScriptedVLM(),
            RecordingEditor(),
            RecordingRetriever(),
            emotion_classifier=None,
        )
        mismatched = instruction_payload(
            element_type="background",
            method="background_change",
            operation_target="carousel",
            instruction="Add a carousel in the background to the right of the balloon.",
        )
        plan = plan_payload()
        plan["instructions"] = [mismatched]

        validated = pipeline._validate_plans(
            {"target_emotion": "amusement", "plans": [plan]},
            analysis_payload(),
            "amusement",
        )

        repaired = validated["plans"][0]["instructions"][0]
        self.assertEqual(repaired["method"], "add")
        self.assertEqual(repaired["element_type"], "object")
        self.assertEqual(repaired["taxonomy_repair"]["original_method"], "background_change")

    def test_reused_distinctive_cue_is_reported_without_plan_generation_failure(self):
        pipeline = EmoAgentPipeline(
            PipelineConfig(output_root=Path("unused"), num_plans=2, steps_per_plan=1),
            ScriptedVLM(),
            RecordingEditor(),
            RecordingRetriever(),
            emotion_classifier=None,
        )
        first = plan_payload()
        second = {
            **plan_payload(),
            "strategy_summary": "a separate floating-bubble treatment",
            "strategy_anchor": "floating bubbles in open sky",
            "distinctive_cues": ["right-background carousel"],
            "instructions": [
                instruction_payload(
                    operation_target="soap bubbles",
                    target_region="the open sky away from the balloon silhouette",
                    instruction="Add translucent soap bubbles to the open sky without overlap.",
                )
            ],
        }

        validated = pipeline._validate_plans(
            {"target_emotion": "amusement", "plans": [first, second]},
            analysis_payload(),
            "amusement",
        )

        self.assertTrue(
            any(
                "Distinctive-cue metadata is reused" in warning
                for warning in validated["diversity_report"]["warnings"]
            )
        )

    def test_attempt_006_recoverable_issues_validate_in_one_pass(self):
        pipeline = EmoAgentPipeline(
            PipelineConfig(output_root=Path("unused"), num_plans=5, steps_per_plan=4),
            ScriptedVLM(),
            RecordingEditor(),
            RecordingRetriever(),
            emotion_classifier=None,
        )
        analysis = analysis_payload()
        analysis["protected_entities"].append(
            {
                "entity": "newspaper",
                "approximate_count": "one",
                "preserve_identity": ["same newspaper"],
                "preserve_geometry": ["same location and folded outline"],
                "immutable_attributes": ["identity and folded silhouette"],
                "editable_attributes": [],
            }
        )
        shared_specs = (
            ("pinwheel", "Add several pinwheels beside the existing path."),
            ("bunting", "Suspend triangular bunting between two background trees."),
            ("soap bubbles", "Add translucent soap bubbles in the open midground."),
        )
        unique_specs = (
            ("carousel", "Add one compact carousel beyond the path."),
            ("kite", "Place one diamond kite high in the open sky."),
            ("comic newspaper print", "Change only the existing newspaper print into comic panels."),
            ("paper lantern", "Introduce one small paper lantern in an open background region."),
            ("colorful windsock", "Include one colorful windsock beside the path."),
        )
        plans: list[dict[str, Any]] = []
        for plan_index, (unique_target, unique_text) in enumerate(unique_specs, 1):
            instructions = [
                instruction_payload(
                    step_id=step_id,
                    operation_target=target,
                    target_region=f"a source-compatible open region for {target}",
                    instruction=text,
                )
                for step_id, (target, text) in enumerate(shared_specs, 1)
            ]
            unique_instruction = instruction_payload(
                step_id=4,
                operation_target=("newspaper" if plan_index == 3 else unique_target),
                target_region=f"the localized region for {unique_target}",
                instruction=unique_text,
                element_type=("object_attribute" if plan_index == 3 else "object"),
                method=("attribute_change" if plan_index == 3 else "add"),
            )
            if plan_index == 2:
                instructions[0] = {
                    **instructions[0],
                    "element_type": "background",
                    "method": "background_change",
                }
            instructions.append(unique_instruction)
            plans.append(
                {
                    "strategy_summary": f"candidate visual path {plan_index}",
                    "strategy_anchor": unique_target,
                    "distinctive_cues": ["playful secondary detail"],
                    "instructions": instructions,
                }
            )

        validated = pipeline._validate_plans(
            {"target_emotion": "amusement", "plans": plans}, analysis, "amusement"
        )

        self.assertEqual(len(validated["plans"]), 5)
        self.assertEqual(validated["plans"][1]["instructions"][0]["method"], "add")
        self.assertEqual(
            validated["plans"][2]["instructions"][3]["constraint_review"]["status"],
            "visual_validation_required",
        )
        self.assertTrue(validated["diversity_report"]["warnings"])

    def test_missing_explanatory_plan_fields_are_deterministically_completed(self):
        pipeline = EmoAgentPipeline(
            PipelineConfig(output_root=Path("unused"), num_plans=1, steps_per_plan=1),
            ScriptedVLM(),
            RecordingEditor(),
            RecordingRetriever(),
            emotion_classifier=None,
        )
        plan = {
            "instructions": [
                {
                    "instruction": "Add one compact carousel beyond the existing path.",
                    "element_type": "background",
                    "method": "background_change",
                }
            ]
        }

        validated = pipeline._validate_plans(
            {"target_emotion": "wrong-copied-label", "plans": [plan]},
            analysis_payload(),
            "amusement",
        )

        completed_plan = validated["plans"][0]
        completed_instruction = completed_plan["instructions"][0]
        self.assertTrue(completed_plan["strategy_summary"])
        self.assertTrue(completed_plan["strategy_anchor"])
        self.assertTrue(completed_plan["distinctive_cues"])
        self.assertEqual(completed_instruction["method"], "add")
        self.assertEqual(completed_instruction["element_type"], "object")
        self.assertTrue(completed_instruction["operation_target"])
        self.assertTrue(completed_instruction["target_region"])
        self.assertTrue(completed_instruction["emotion_rationale"])
        self.assertTrue(completed_instruction["preserve"])

    def test_joint_evaluator_schema_failure_falls_back_to_valid_generated_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, _, _, _ = self._run_pipeline(
                root,
                vlm=ScriptedVLM(malformed_joint_evaluation=True),
                max_json_retries=2,
            )

            self.assertEqual(summary["accepted"], 1)
            plans = json.loads(
                (root / "runs/test_run/plans.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plans["joint_evaluation"]["verdict"], "reasonable")
            self.assertIn(
                "retained validated bundle",
                plans["joint_evaluation"]["issues"][0],
            )
            trajectory = (root / "runs/test_run/trajectory.jsonl").read_text(encoding="utf-8")
            self.assertIn("Planning_evaluate_fallback", trajectory)

    def test_mock_end_to_end_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, _, editor, _ = self._run_pipeline(root)
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(len(editor.calls), 1)
            self.assertTrue((root / "runs/test_run/plan_01/final.png").is_file())
            self.assertTrue((root / "runs/test_run/trajectory.jsonl").is_file())

    def test_face_constraint_rejects_face_edit_for_faceless_source(self):
        analysis = {"facial_expression_available": False}
        instruction = {
            "element_type": "facial_expression",
            "method": "expression_change",
            "instruction": "Give the subject an angry scowl.",
        }
        with self.assertRaisesRegex(ValueError, "face-expression editing is forbidden"):
            EmoAgentPipeline._assert_face_constraint(instruction, analysis)

    def test_face_constraint_allows_secondary_person_without_expression_edit(self):
        analysis = {"facial_expression_available": False}
        instruction = {
            "element_type": "object",
            "method": "add",
            "instruction": "Add a person in the distant background.",
        }
        EmoAgentPipeline._assert_face_constraint(instruction, analysis)

    def test_prior_step_failure_is_passed_to_next_editing_prompt(self):
        _, user = flux_prompt_generation(
            analysis=analysis_payload(malicious_source_emotion=None),
            source_emotion_label="contentment",
            target_emotion="fear",
            instruction=instruction_payload(
                element_type="scene",
                method="background_change",
                operation_target="sky weather",
                target_region="the existing sky",
                instruction="Transform the clear sky into an approaching storm.",
                target_emotion="fear",
            ),
            prior_step_failures=[
                {
                    "failed_step_id": 1,
                    "problems": ["the balloon disappeared"],
                    "action_taken": "rejected output rolled back to last verified input",
                }
            ],
        )

        self.assertIn("Earlier rejected edits were rolled back", user)
        self.assertIn("the balloon disappeared", user)
        self.assertIn("verification_spec", user)
        self.assertNotIn("expected_visible_change", user)

    def test_objective_gate_rejects_near_black_fear_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            edited = root / "edited.png"
            Image.new("RGB", (64, 64), (180, 180, 180)).save(source)
            Image.new("RGB", (64, 64), (2, 2, 2)).save(edited)

            checks = EmoAgentPipeline._objective_image_checks(source, edited, "fear")

            self.assertFalse(checks["brightness_ok"])
            self.assertGreater(checks["near_black_fraction"], 0.9)

    def test_batch_selector_prefers_critic_acceptance_and_compacts_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = root / "accepted.png"
            unresolved = root / "unresolved.png"
            Image.new("RGB", (32, 32), "red").save(accepted)
            Image.new("RGB", (32, 32), "blue").save(unresolved)
            summary = {
                "results": [
                    {
                        "plan_id": "plan_01",
                        "status": "accepted",
                        "preservation_gate_passed": True,
                        "final_image": str(accepted),
                        "final_probabilities": {"sadness": 0.4},
                    },
                    {
                        "plan_id": "plan_02",
                        "status": "unresolved",
                        "preservation_gate_passed": True,
                        "final_image": str(unresolved),
                        "final_probabilities": {"sadness": 0.9},
                    },
                ]
            }
            selection = select_final(summary, "sadness")
            self.assertEqual(selection["selected"]["plan_id"], "plan_01")
            compacted = compact_attempt_images(root)
            self.assertEqual(compacted["removed_png_count"], 2)
            self.assertFalse(accepted.exists())

    def test_selector_falls_back_to_source_when_all_candidates_are_unsafe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            unsafe = root / "unsafe.png"
            Image.new("RGB", (32, 32), "blue").save(source)
            Image.new("RGB", (32, 32), "black").save(unsafe)
            summary = {
                "source_image": str(source),
                "source_probabilities": {"fear": 0.1},
                "results": [
                    {
                        "plan_id": "plan_01",
                        "status": "accepted",
                        "preservation_gate_passed": False,
                        "final_image": str(unsafe),
                        "final_probabilities": {"fear": 0.99},
                    }
                ],
            }

            selection = select_final(summary, "fear")

            self.assertTrue(selection["source_fallback_used"])
            self.assertEqual(selection["selected"]["plan_id"], "source_fallback")
            self.assertEqual(selection["selected"]["final_image"], str(source))

    def test_metric_compaction_keeps_source_and_plan_finals(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_001"
            final = attempt / "plan_01/final.png"
            step = attempt / "plan_01/pre_creation/step_01/output_try_01.png"
            final.parent.mkdir(parents=True)
            step.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16), "white").save(attempt / "source.png")
            Image.new("RGB", (16, 16), "red").save(final)
            Image.new("RGB", (16, 16), "blue").save(step)

            compacted = compact_attempt_images(attempt, retain_metric_images=True)

            self.assertTrue((attempt / "source.png").is_file())
            self.assertTrue(final.is_file())
            self.assertFalse(step.exists())
            self.assertEqual(compacted["retained_png_count"], 2)
            self.assertEqual(compacted["removed_png_count"], 1)

    def test_random_assignments_are_persisted_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [root / "1.png", root / "2.png", root / "duplicate.png"]
            emotions = ["amusement", "awe", "sadness"]
            assignment_path = root / "process/emotion_assignments.json"

            first = load_or_create_random_assignments(images, emotions, assignment_path, seed=2024)
            second = load_or_create_random_assignments(images, emotions, assignment_path, seed=9999)

            self.assertEqual(first, second)
            self.assertEqual(set(first), {path.name for path in images})
            self.assertTrue(set(first.values()).issubset(set(emotions)))

    def test_unlabeled_recursive_discovery_and_seeded_random_sample_of_ten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            for index in range(15):
                path = input_dir / ("nested" if index % 2 else "") / f"{index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), "white").save(path)
            generated = input_dir / "process/ignored.png"
            generated.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "black").save(generated)

            discovered = discover_source_images(input_dir)
            first = sample_source_images(discovered, 10, seed=2024)
            second = sample_source_images(discovered, 10, seed=2024)

            self.assertEqual(len(discovered), 15)
            self.assertEqual([item.assignment_key for item in first], [item.assignment_key for item in second])
            self.assertEqual(len(first), 10)
            self.assertNotIn("process/ignored.png", {item.assignment_key for item in discovered})

    def test_batch_pair_writes_process_and_vlm_source_emotion_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "inputs/contentment/100.png"
            source.parent.mkdir(parents=True)
            Image.new("RGB", (640, 480), "blue").save(source)
            args = Namespace(
                process_root=root / "process",
                result_root=root / "result",
                force=False,
                pair_attempts=1,
                num_plans=1,
                steps_per_plan=1,
                top_k=5,
                max_json_retries=2,
                max_step_retries=0,
                max_critic_rounds=0,
                seed=2024,
                require_resnet_target=False,
                keep_intermediate_images=False,
                retain_metric_images=True,
            )
            ok, outcome = run_pair(
                args,
                source,
                image_index=1,
                emotion="sadness",
                emotion_index=0,
                backends=(ScriptedVLM(), RecordingEditor(), RecordingRetriever(), None),
            )
            self.assertTrue(ok)
            self.assertEqual(outcome, "completed")
            self.assertTrue((root / "result/100/sadness.png").is_file())
            status = json.loads(
                (root / "process/100/sadness/status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["source_emotion"], "anger")
            self.assertEqual(
                status["source_emotion_provenance"], "planning_agent_vlm_prediction"
            )
            self.assertEqual(status["target_emotion"], "sadness")


if __name__ == "__main__":
    unittest.main()
