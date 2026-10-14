from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .backends import (
    ImageEditorBackend,
    ResNetEmotionClassifier,
    VisionLanguageBackend,
    empty_emotion_probabilities,
)
from .constants import EDIT_METHODS, ELEMENT_TYPES, EMOTIONS
from .io_utils import append_jsonl, copy_image, prepare_source_image, require_keys, write_json
from .prompts import (
    critic_prompt,
    flux_prompt_generation,
    plan_generation_prompt,
    planning_evaluation_prompt,
    retrieval_query,
    source_analysis_prompt,
    step_validation_prompt,
)


PIPELINE_REVISION = "single-plan-v6-first-prompt-relaxed-emotion-edit"


@dataclass
class PipelineConfig:
    output_root: Path
    num_plans: int = 5
    steps_per_plan: int = 4
    top_k: int = 5
    max_json_retries: int = 3
    max_step_retries: int = 2
    max_critic_rounds: int = 2
    seed: int = 2024
    image_size: int = 512
    require_resnet_target: bool = False


class EmoAgentPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        vlm: VisionLanguageBackend,
        editor: ImageEditorBackend,
        retriever: Any,
        emotion_classifier: ResNetEmotionClassifier | None,
    ) -> None:
        self.config = config
        self.vlm = vlm
        self.editor = editor
        self.retriever = retriever
        self.classifier = emotion_classifier
        self.run_dir: Path | None = None
        self.trajectory_path: Path | None = None

    def _event(self, agent: str, action: str, action_input: Any, observation: Any) -> None:
        if self.trajectory_path is None:
            raise RuntimeError("pipeline run directory is not initialized")
        append_jsonl(
            self.trajectory_path,
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "agent": agent,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            },
        )

    def _ask_json(
        self,
        stage: str,
        agent: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None,
        validator: Callable[[dict[str, Any]], dict[str, Any]],
        max_new_tokens: int = 1024,
    ) -> dict[str, Any]:
        errors: list[str] = []
        for attempt in range(1, self.config.max_json_retries + 1):
            corrected_user = user_prompt
            if errors:
                corrected_user += (
                    "\n\nYour previous response failed schema validation: "
                    + errors[-1]
                    + "\nReturn a corrected JSON object only."
                )
            try:
                raw, value = self.vlm.generate_json(
                    system_prompt,
                    corrected_user,
                    image_paths=image_paths,
                    max_new_tokens=max_new_tokens,
                )
                validated = validator(value)
                self._event(
                    agent,
                    stage,
                    {
                        "attempt": attempt,
                        "system_prompt": system_prompt,
                        "user_prompt": corrected_user,
                        "images": [str(p) for p in image_paths or []],
                    },
                    {"status": "ok", "raw_response": raw, "parsed": validated},
                )
                return validated
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                self._event(
                    agent,
                    stage,
                    {
                        "attempt": attempt,
                        "system_prompt": system_prompt,
                        "user_prompt": corrected_user,
                        "images": [str(p) for p in image_paths or []],
                    },
                    {"status": "retry", "error": errors[-1]},
                )
        raise RuntimeError(f"{stage} failed after {self.config.max_json_retries} attempts: {errors}")

    @staticmethod
    def _validate_analysis(value: dict[str, Any]) -> dict[str, Any]:
        require_keys(
            value,
            (
                "main_subject",
                "objects",
                "scene",
                "composition",
                "object_attributes",
                "expressions_actions",
                "visible_people_or_faces",
                "facial_expression_available",
                "protected_entities",
                "must_preserve_layout",
                "editable_or_replaceable_elements",
                "key_information_rationale",
                "lighting",
                "global_hue",
                "source_emotion",
                "semantic_summary",
                "retrieval_semantics",
            ),
            "source analysis",
        )
        affective_interpretation = re.compile(
            r"\b(?:amusement|awe|contentment|excitement|anger|disgust|fear|sadness|"
            r"excited|exciting|joyful|cheerful|sad|angry|fearful|frightening|"
            r"amused|amusing|contented|disgusted|ominous|gloomy|anticipation|"
            r"festive|celebratory|playful|serene|peaceful|lively|vibrant)\b",
            re.IGNORECASE,
        )
        semantic_summary = str(value["semantic_summary"]).strip()
        if not semantic_summary:
            raise ValueError("semantic_summary is empty")
        retrieval_semantics = str(value["retrieval_semantics"]).strip()
        if not retrieval_semantics:
            raise ValueError("retrieval_semantics is empty")
        if affective_interpretation.search(retrieval_semantics):
            # Keep the retrieval query usable without discarding an otherwise
            # valid source analysis. Prefer the concrete summary after removing
            # affect words; the target emotion is still never inserted here.
            neutralized = affective_interpretation.sub("", retrieval_semantics)
            neutralized = re.sub(r"\s+", " ", neutralized).strip(" ,.;:-")
            if len(neutralized.split()) < 3:
                neutralized = affective_interpretation.sub("", semantic_summary)
                neutralized = re.sub(r"\s+", " ", neutralized).strip(" ,.;:-")
            if not neutralized:
                raise ValueError(
                    "retrieval_semantics contains only inferred affect and cannot be neutralized"
                )
            value["retrieval_semantics"] = neutralized
            value["retrieval_semantics_normalized"] = True
        else:
            value["retrieval_semantics"] = retrieval_semantics
            value["retrieval_semantics_normalized"] = False
        if not isinstance(value["visible_people_or_faces"], list):
            raise ValueError("visible_people_or_faces must be a list")
        if not isinstance(value["facial_expression_available"], bool):
            raise ValueError("facial_expression_available must be a boolean")
        protected = value["protected_entities"]
        if not isinstance(protected, list) or not protected:
            raise ValueError("protected_entities must be a non-empty list")
        entity_names: list[str] = []
        for entity in protected:
            if not isinstance(entity, dict):
                raise ValueError("each protected entity must be an object")
            require_keys(
                entity,
                (
                    "entity",
                    "approximate_count",
                    "preserve_identity",
                    "preserve_geometry",
                    "immutable_attributes",
                    "editable_attributes",
                ),
                "protected entity",
            )
            name = str(entity["entity"]).strip()
            if not name:
                raise ValueError("protected entity name is empty")
            entity_names.append(name)
            for field in (
                "preserve_identity",
                "preserve_geometry",
                "immutable_attributes",
                "editable_attributes",
            ):
                if not isinstance(entity[field], list):
                    raise ValueError(f"protected entity {field} must be a list")
            immutable = {str(item).strip().casefold() for item in entity["immutable_attributes"]}
            editable = {str(item).strip().casefold() for item in entity["editable_attributes"]}
            overlap = (immutable & editable) - {""}
            if overlap:
                raise ValueError(
                    f"protected entity has attributes marked both immutable and editable: {sorted(overlap)}"
                )
        if not isinstance(value["must_preserve_layout"], list) or not value["must_preserve_layout"]:
            raise ValueError("must_preserve_layout must be a non-empty list")
        if not isinstance(value["editable_or_replaceable_elements"], list):
            raise ValueError("editable_or_replaceable_elements must be a list")
        if not str(value["key_information_rationale"]).strip():
            raise ValueError("key_information_rationale is empty")
        source_emotion = str(value["source_emotion"]).strip().casefold()
        if source_emotion not in set(EMOTIONS) | {"mixed/uncertain"}:
            raise ValueError(
                "source_emotion must be one of the eight emotions or mixed/uncertain"
            )
        evidence = value.get("source_emotion_evidence")
        if isinstance(evidence, str) and evidence.strip():
            evidence = [evidence.strip()]
        elif not isinstance(evidence, list):
            evidence = []
        evidence = [str(item).strip() for item in evidence if str(item).strip()]
        if not evidence:
            visible_fields = [
                str(value.get("expressions_actions") or "").strip(),
                str(value.get("lighting") or "").strip(),
                str(value.get("global_hue") or "").strip(),
            ]
            evidence = [
                "Source-emotion prediction is based on the visible actions, lighting, and color cues: "
                + "; ".join(item for item in visible_fields if item)
            ]
        value["source_emotion_evidence"] = evidence
        value["source_emotion"] = source_emotion
        # Compatibility for existing result consumers.  This derived list means
        # entity existence/identity, not that every entity attribute is immutable.
        value["must_preserve_entities"] = entity_names
        return value

    @staticmethod
    def _assert_face_constraint(instruction: dict[str, Any], analysis: dict[str, Any]) -> None:
        """Prevent a missing face from becoming an editable emotion-bearing object."""
        if bool(analysis.get("facial_expression_available")):
            return
        element = str(instruction.get("element_type", "")).casefold()
        method = str(instruction.get("method", "")).casefold()
        text = str(instruction.get("instruction", "")).casefold()
        if element == "facial_expression" or method == "expression_change":
            raise ValueError(
                "facial expression is unavailable in the source; face-expression editing is forbidden"
            )
        if re.search(r"\b(?:frown|scowl|angry face|smiling face|facial expression)\b", text):
            raise ValueError(
                "facial expression is unavailable in the source; requesting a face expression is forbidden"
            )

    @classmethod
    def _repair_face_instruction(
        cls,
        instruction: dict[str, Any],
        analysis: dict[str, Any],
        target_emotion: str,
        step_index: int,
    ) -> dict[str, Any]:
        """Deterministically replace an illegal face edit without adding new scene subjects."""
        try:
            cls._assert_face_constraint(instruction, analysis)
            return instruction
        except ValueError as exc:
            moods = {
                "amusement": "brighter, warm, gently playful",
                "awe": "dramatic, luminous, and visually expansive",
                "contentment": "soft, balanced, warm, and tranquil",
                "excitement": "vivid, high-energy, and strongly directional",
                "anger": "harsh, tense, high-contrast, and red-orange",
                "disgust": "sickly, muted, uneasy, and greenish",
                "fear": "cold, tense, shadowed, while retaining readable midtones",
                "sadness": "overcast, subdued, cool, and low-contrast",
            }
            mood = moods.get(target_emotion, f"visually expressive of {target_emotion}")
            original = {
                "element_type": instruction.get("element_type"),
                "method": instruction.get("method"),
                "instruction": instruction.get("instruction"),
            }
            preserve = list(instruction.get("preserve") or [])
            for item in (
                "all existing subjects and objects",
                "absence of new people or faces",
                "identity",
                "composition",
            ):
                if item not in preserve:
                    preserve.append(item)
            repaired = {
                **instruction,
                "step_id": step_index,
                "element_type": "color_tone",
                "method": "filter_change",
                "operation_target": "existing scene lighting and color treatment",
                "target_region": "the entire existing frame without changing local geometry",
                "instruction": (
                    f"Change only the existing scene lighting and color treatment to feel {mood}; "
                    "preserve every existing subject and object and keep the scene free of new people, "
                    "faces, portraits, characters, and facial expressions."
                ),
                "emotion_rationale": (
                    f"Conveys {target_emotion} through the source scene's existing lighting and color "
                    "because no editable facial expression is visible."
                ),
                "preserve": preserve,
                "constraint_repair": {
                    "reason": str(exc),
                    "original_instruction": original,
                    "policy": "replace unavailable facial-expression factor with a non-person scene factor",
                },
            }
            cls._assert_face_constraint(repaired, analysis)
            return repaired

    @classmethod
    def _normalize_instruction_constraints(
        cls,
        instruction: dict[str, Any],
        analysis: dict[str, Any],
        target_emotion: str,
        step_index: int,
    ) -> dict[str, Any]:
        """Apply deterministic repairs for constraints that do not need another VLM call."""
        # Normalize aliases first so an expression alias cannot bypass the
        # faceless-source safety check.  Face repair emits canonical taxonomy.
        repaired = cls._repair_instruction_taxonomy(instruction)
        repaired = cls._repair_face_instruction(
            repaired, analysis, target_emotion, step_index
        )
        cls._validate_instruction_taxonomy(repaired)
        cls._assert_face_constraint(repaired, analysis)
        return repaired

    @staticmethod
    def _normalize_plan_text(value: Any) -> str:
        text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold())
        ignored = {
            "a", "an", "the", "to", "of", "in", "on", "at", "and", "with",
            "left", "right", "side", "slightly", "small", "large",
        }
        return " ".join(token for token in text.split() if token not in ignored)

    @classmethod
    def _instruction_signature(cls, instruction: dict[str, Any]) -> tuple[str, str]:
        # Region, wording and element label are deliberately excluded: changing
        # only a side, paraphrase, or inconsistent taxonomy cannot fake diversity.
        return (
            cls._normalize_plan_text(instruction.get("method")),
            cls._normalize_plan_text(instruction.get("operation_target")),
        )

    @classmethod
    def _instructions_equivalent(
        cls, left: dict[str, Any], right: dict[str, Any]
    ) -> bool:
        left_signature = cls._instruction_signature(left)
        right_signature = cls._instruction_signature(right)
        if left_signature == right_signature:
            return True
        left_tokens = set(cls._normalize_plan_text(left.get("instruction")).split())
        right_tokens = set(cls._normalize_plan_text(right.get("instruction")).split())
        if not left_tokens or not right_tokens:
            return False
        lexical_overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        return left_signature[0] == right_signature[0] and lexical_overlap >= 0.72

    @classmethod
    def _shared_instruction_count(
        cls, left: list[dict[str, Any]], right: list[dict[str, Any]]
    ) -> int:
        unmatched = set(range(len(right)))
        shared = 0
        for instruction in left:
            for index in list(unmatched):
                if cls._instructions_equivalent(instruction, right[index]):
                    unmatched.remove(index)
                    shared += 1
                    break
        return shared

    @staticmethod
    def _validate_instruction_taxonomy(instruction: dict[str, Any]) -> None:
        element = str(instruction["element_type"]).casefold()
        method = str(instruction["method"]).casefold()
        if element not in set(ELEMENT_TYPES) | {"object_attribute", "background"}:
            raise ValueError(f"unsupported element_type: {element}")
        if method not in EDIT_METHODS:
            raise ValueError(f"unsupported editing method: {method}")
        text = str(instruction["instruction"]).strip().casefold()
        if not text:
            raise ValueError("instruction text is empty")
        if not str(instruction["emotion_rationale"]).strip():
            raise ValueError("emotion_rationale is empty")
        add_like = bool(
            re.search(
                r"\b(?:add|place|insert|introduce|include|increase (?:the )?(?:number|count))\b",
                text,
            )
        )
        attribute_addition = element in {"object_attribute", "color_tone"} and bool(
            re.search(
                r"\b(?:pattern|stripe|texture|color|colour|tint|glow|highlight|shadow|lighting)\b",
                text,
            )
        )
        background_atmosphere = element in {"scene", "background"} and bool(
            re.search(r"\b(?:sky|cloud|weather|rain|snow|fog|mist|storm)\b", text)
        )
        addition_exception = attribute_addition or (
            method == "background_change" and background_atmosphere
        )
        if add_like and not addition_exception and method != "add":
            raise ValueError(
                "an addition/count increase must use method=add; depth location does not make it background_change"
            )
        if method == "background_change" and add_like and not addition_exception:
            raise ValueError("background_change transforms existing background; it cannot label an addition")

    @classmethod
    def _repair_instruction_taxonomy(cls, instruction: dict[str, Any]) -> dict[str, Any]:
        """Normalize common label mistakes while preserving the requested visible operation.

        Taxonomy labels are routing metadata.  When the imperative clearly says
        to add an entity, retrying the VLM because it wrote
        ``background_change`` is wasteful and can make a whole batch fail.
        """
        repaired = dict(instruction)
        original_element = str(repaired.get("element_type", "")).strip().casefold()
        original_method = str(repaired.get("method", "")).strip().casefold()
        element_aliases = {
            "attribute": "object_attribute",
            "object attributes": "object_attribute",
            "object_attributes": "object_attribute",
            "colour_tone": "color_tone",
            "color tone": "color_tone",
            "global_hue": "color_tone",
            "expression": "facial_expression",
            "facial expression": "facial_expression",
        }
        method_aliases = {
            "addition": "add",
            "adding": "add",
            "removal": "remove",
            "deletion": "remove",
            "replacement": "replace",
            "attribute": "attribute_change",
            "change_attribute": "attribute_change",
            "filter": "filter_change",
            "background": "background_change",
            "expression": "expression_change",
        }
        element = element_aliases.get(original_element, original_element)
        method = method_aliases.get(original_method, original_method)
        text = str(repaired.get("instruction", "")).strip().casefold()
        add_like = bool(
            re.search(
                r"\b(?:add|place|insert|introduce|include|increase (?:the )?(?:number|count))\b",
                text,
            )
        )
        attribute_addition = element in {"object_attribute", "color_tone"} and bool(
            re.search(
                r"\b(?:pattern|stripe|texture|color|colour|tint|glow|highlight|shadow|lighting)\b",
                text,
            )
        )
        background_atmosphere = element in {"scene", "background"} and bool(
            re.search(r"\b(?:sky|cloud|weather|rain|snow|fog|mist|storm)\b", text)
        )
        if add_like and not attribute_addition and not (
            method == "background_change" and background_atmosphere
        ):
            method = "add"
            if element == "background":
                element = "object"
        if method not in EDIT_METHODS:
            if add_like:
                method = "add"
            elif re.search(r"\b(?:remove|delete|erase)\b", text):
                method = "remove"
            elif re.search(r"\b(?:replace|swap)\b", text):
                method = "replace"
            elif re.search(r"\b(?:light|lighting|color|colour|tone|hue|filter)\b", text):
                method = "filter_change"
            else:
                method = "attribute_change"
        allowed_elements = set(ELEMENT_TYPES) | {"object_attribute", "background"}
        if element not in allowed_elements:
            element = {
                "expression_change": "facial_expression",
                "filter_change": "color_tone",
                "background_change": "scene",
                "attribute_change": "object_attribute",
            }.get(method, "object")
        repaired["element_type"] = element
        repaired["method"] = method
        if (element, method) != (original_element, original_method):
            repaired["taxonomy_repair"] = {
                "original_element_type": original_element,
                "original_method": original_method,
                "normalized_element_type": element,
                "normalized_method": method,
                "policy": "deterministic normalization from the visible imperative",
            }
        cls._validate_instruction_taxonomy(repaired)
        return repaired

    @staticmethod
    def _infer_operation_target(instruction_text: str) -> str:
        text = re.sub(r"\s+", " ", str(instruction_text)).strip()
        match = re.search(
            r"\b(?:add|place|insert|introduce|include|increase|change|modify|shift|"
            r"adjust|remove|delete|erase|replace|swap)\s+(?:only\s+)?"
            r"(?:the\s+|a\s+|an\s+)?(.+?)"
            r"(?=\s+(?:in|on|at|to|into|from|beside|behind|above|below|while|"
            r"without|but|so that)\b|[.;,]|$)",
            text,
            re.IGNORECASE,
        )
        if match and match.group(1).strip():
            return match.group(1).strip()
        words = re.findall(r"[A-Za-z0-9'-]+", text)
        return " ".join(words[:12]) or "localized visible source content"

    def _validate_plans(
        self,
        value: dict[str, Any],
        analysis: dict[str, Any],
        target_emotion: str,
    ) -> dict[str, Any]:
        # The CLI target is authoritative.  A copied/wrong model metadata value
        # must not waste a complete planning call or change the requested task.
        supplied_target = str(value.get("target_emotion", "")).strip().casefold()
        plans = value.get("plans")
        if not isinstance(plans, list) or len(plans) != self.config.num_plans:
            raise ValueError(f"plans must contain exactly {self.config.num_plans} items")
        normalized: list[dict[str, Any]] = []
        summaries: set[str] = set()
        cue_owners: dict[str, str] = {}
        bundle_warnings: list[str] = []
        if supplied_target != target_emotion:
            bundle_warnings.append(
                f"Model returned target_emotion={supplied_target!r}; normalized to authoritative "
                f"target {target_emotion!r}."
            )
        for plan_index, plan in enumerate(plans, 1):
            if not isinstance(plan, dict):
                raise ValueError("every plan must be an object")
            if "instructions" not in plan:
                raise ValueError("plan is missing executable instructions")
            summary = str(plan.get("strategy_summary", "")).strip()
            if not summary:
                summary = str(plan.get("strategy_anchor", "")).strip() or f"visual strategy {plan_index}"
                bundle_warnings.append(
                    f"plan_{plan_index:02d} had an empty strategy_summary; a fallback summary was recorded."
                )
            if summary.casefold() in summaries:
                bundle_warnings.append(
                    f"plan_{plan_index:02d} reuses a strategy summary; actual instructions determine diversity."
                )
            summaries.add(summary.casefold())
            strategy_anchor = str(plan.get("strategy_anchor", "")).strip()
            if not strategy_anchor:
                strategy_anchor = summary
                bundle_warnings.append(
                    f"plan_{plan_index:02d} had an empty strategy_anchor; its summary was used."
                )
            cues = plan.get("distinctive_cues", [])
            # One genuinely distinctive anchor is sufficient for an independent
            # path.  Requiring two exclusive cues per plan made valid bundles
            # brittle when several source-compatible preservation/support edits
            # were necessarily shared.
            minimum_cues = 1
            if isinstance(cues, str):
                cues = [cues]
            if not isinstance(cues, list):
                cues = []
            cues = [str(cue).strip() for cue in cues if str(cue).strip()]
            if len(cues) < minimum_cues:
                cues = [strategy_anchor]
                bundle_warnings.append(
                    f"plan_{plan_index:02d} lacked a usable distinctive cue; its strategy anchor was used."
                )
            for cue in cues:
                normalized_cue = self._normalize_plan_text(cue)
                if normalized_cue in cue_owners:
                    bundle_warnings.append(
                        f"Distinctive-cue metadata is reused by {cue_owners[normalized_cue]} and "
                        f"plan_{plan_index:02d}; actual instruction sets determine diversity."
                    )
                else:
                    cue_owners[normalized_cue] = f"plan_{plan_index:02d}"
            instructions = plan["instructions"]
            if not isinstance(instructions, list) or len(instructions) != self.config.steps_per_plan:
                raise ValueError(
                    f"each plan must contain exactly {self.config.steps_per_plan} instructions"
                )
            normalized_instructions: list[dict[str, Any]] = []
            for step_index, instruction in enumerate(instructions, 1):
                if not isinstance(instruction, dict):
                    raise ValueError("every instruction must be an object")
                instruction_text = str(instruction.get("instruction", "")).strip()
                if not instruction_text:
                    raise ValueError("instruction text is empty and cannot be executed")
                instruction = dict(instruction)
                instruction["instruction"] = instruction_text
                if not str(instruction.get("operation_target", "")).strip():
                    instruction["operation_target"] = self._infer_operation_target(instruction_text)
                    bundle_warnings.append(
                        f"plan_{plan_index:02d} step_{step_index:02d} lacked operation_target; "
                        "one was derived from the imperative."
                    )
                if not str(instruction.get("target_region", "")).strip():
                    instruction["target_region"] = (
                        "the localized source-compatible region described by this instruction"
                    )
                    bundle_warnings.append(
                        f"plan_{plan_index:02d} step_{step_index:02d} lacked target_region; "
                        "the Editing Agent must localize it from the current image."
                    )
                if not str(instruction.get("emotion_rationale", "")).strip():
                    instruction["emotion_rationale"] = (
                        f"The requested visible change is intended to contribute to {target_emotion}; "
                        "the step validator and final Critic must verify that contribution."
                    )
                instruction.setdefault("element_type", "")
                instruction.setdefault("method", "")
                instruction.setdefault("preserve", [])
                repaired = self._normalize_instruction_constraints(
                    instruction, analysis, target_emotion, step_index
                )
                if not str(repaired["operation_target"]).strip():
                    raise ValueError("instruction operation_target is empty")
                if not str(repaired["target_region"]).strip():
                    raise ValueError("instruction target_region is empty")
                if isinstance(repaired["preserve"], str):
                    repaired["preserve"] = [repaired["preserve"]]
                repaired["preserve"] = [
                    str(item).strip()
                    for item in (repaired.get("preserve") or [])
                    if str(item).strip()
                ]
                if not repaired["preserve"]:
                    repaired["preserve"] = [
                        "keep the main subject recognizable",
                        "keep the broad source scene coherent",
                    ]
                    bundle_warnings.append(
                        f"plan_{plan_index:02d} step_{step_index:02d} lacked preserve constraints; "
                        "they were deterministically derived from source analysis."
                    )
                self._validate_instruction_taxonomy(repaired)
                explicit_target = self._normalize_plan_text(repaired["operation_target"])
                explicit_target_tokens = set(explicit_target.split())
                protected_names = {
                    self._normalize_plan_text(entity["entity"]): entity
                    for entity in analysis.get("protected_entities", [])
                    if isinstance(entity, dict) and entity.get("entity")
                }
                constraint_warnings: list[str] = []
                for protected_name, entity in protected_names.items():
                    protected_tokens = set(protected_name.split())
                    targets_protected_entity = bool(
                        protected_tokens
                        and explicit_target_tokens
                        and (
                            protected_tokens <= explicit_target_tokens
                            or explicit_target_tokens <= protected_tokens
                        )
                    )
                    if targets_protected_entity:
                        if repaired["method"] in {"remove", "replace"}:
                            original_operation = {
                                "method": repaired["method"],
                                "instruction": repaired["instruction"],
                            }
                            repaired = {
                                **repaired,
                                "element_type": "object_attribute",
                                "method": "attribute_change",
                                "operation_target": str(entity["entity"]),
                                "instruction": (
                                    f"Change only a small non-identity surface detail of the existing "
                                    f"{entity['entity']} to provide a subtle visual cue supporting "
                                    f"{target_emotion}; preserve its identity, count, position, scale, "
                                    "silhouette, geometry, and every immutable attribute."
                                ),
                                "constraint_repair": {
                                    "reason": "destructive operation targeted a protected entity",
                                    "original_operation": original_operation,
                                    "policy": "convert to a reversible non-identity attribute treatment",
                                },
                            }
                            constraint_warnings.append(
                                f"A destructive operation targeting protected entity {entity['entity']!r} "
                                "was converted to a non-identity attribute treatment."
                            )
                        editable_text = self._normalize_plan_text(
                            " ".join(map(str, entity.get("editable_attributes") or []))
                        )
                        editable_tokens = set(editable_text.split())
                        instruction_tokens = set(
                            self._normalize_plan_text(
                                f"{repaired['operation_target']} {repaired['instruction']}"
                            ).split()
                        )
                        if repaired["method"] == "attribute_change" and not any(
                            editable in instruction_tokens
                            or any(
                                token.startswith(editable) or editable.startswith(token)
                                for token in instruction_tokens
                                if len(token) >= 4 and len(editable) >= 4
                            )
                            for editable in editable_tokens
                        ):
                            constraint_warnings.append(
                                f"The requested attribute change to protected entity "
                                f"{entity['entity']!r} was not lexically matched to the "
                                "non-exhaustive editable_attributes hints; require step-level "
                                "visual verification of identity, geometry, immutable attributes, "
                                "and all unrequested properties."
                            )
                        if repaired["method"] == "add" and not (
                            editable_tokens & {"count", "number", "quantity", "density"}
                        ):
                            constraint_warnings.append(
                                f"The requested addition involving protected entity "
                                f"{entity['entity']!r} was not lexically matched to a count hint; "
                                "require step-level visual verification of the protected entity's "
                                "identity, geometry, and approximate count."
                            )
                if constraint_warnings:
                    repaired = {
                        **repaired,
                        "constraint_review": {
                            "status": "visual_validation_required",
                            "warnings": constraint_warnings,
                        },
                    }
                normalized_instructions.append({**repaired, "step_id": step_index})
            for left_index, left in enumerate(normalized_instructions):
                if any(
                    self._instructions_equivalent(left, right)
                    for right in normalized_instructions[left_index + 1 :]
                ):
                    bundle_warnings.append(
                        f"plan_{plan_index:02d} contains redundant equivalent instructions; "
                        "the joint evaluator should revise them and the Editing Agent may mark an "
                        "already-satisfied repeat as a no-op."
                    )
            normalized.append(
                {
                    **plan,
                    "plan_id": f"plan_{plan_index:02d}",
                    "strategy_summary": summary,
                    "strategy_anchor": strategy_anchor,
                    "distinctive_cues": cues,
                    "instructions": normalized_instructions,
                }
            )
        recommended_max_shared = self.config.steps_per_plan // 2
        pairwise_diversity: list[dict[str, Any]] = []
        diversity_warnings: list[str] = []
        for left_index, left in enumerate(normalized):
            left_set = frozenset(
                self._instruction_signature(item) for item in left["instructions"]
            )
            for right in normalized[left_index + 1 :]:
                right_set = frozenset(
                    self._instruction_signature(item) for item in right["instructions"]
                )
                if left_set == right_set:
                    bundle_warnings.append(
                        f"{left['plan_id']} and {right['plan_id']} have the same order-insensitive "
                        "operation set; the joint evaluator should revise at least one operation."
                    )
                shared = self._shared_instruction_count(
                    left["instructions"], right["instructions"]
                )
                # A complete semantic duplicate remains invalid.  Partial
                # overlap is reported rather than rejected: plans often need
                # common, source-compatible support edits, and one meaningful
                # distinct operation can still yield an independent result.
                if shared >= self.config.steps_per_plan:
                    bundle_warnings.append(
                        f"{left['plan_id']} and {right['plan_id']} are complete semantic duplicates; "
                        "accepted for recovery but marked high risk for joint evaluation and Critic review."
                    )
                warning = shared > recommended_max_shared
                pairwise_diversity.append(
                    {
                        "left_plan_id": left["plan_id"],
                        "right_plan_id": right["plan_id"],
                        "shared_operations": shared,
                        "total_operations": self.config.steps_per_plan,
                        "recommended_max_shared": recommended_max_shared,
                        "warning": warning,
                    }
                )
                if warning:
                    diversity_warnings.append(
                        f"{left['plan_id']} and {right['plan_id']} share {shared}/"
                        f"{self.config.steps_per_plan} operations; accepted because at least "
                        "one meaningful operation differs."
                    )
        return {
            "target_emotion": target_emotion,
            "plans": normalized,
            "diversity_report": {
                "policy": (
                    "Reject complete order-insensitive semantic duplicates; report partial "
                    "overlap above the recommended threshold without aborting inference."
                ),
                "pairwise": pairwise_diversity,
                "warnings": bundle_warnings + diversity_warnings,
            },
        }

    def _validate_plan_evaluation(
        self,
        value: dict[str, Any],
        analysis: dict[str, Any],
        target_emotion: str,
    ) -> dict[str, Any]:
        if str(value.get("verdict", "")).lower() not in {"reasonable", "revise"}:
            raise ValueError("planning verdict must be reasonable or revise")
        issues = value.get("issues", [])
        if not isinstance(issues, list):
            raise ValueError("planning issues must be a list")
        revised_plans = value.get("revised_plans")
        validated = self._validate_plans(
            {"target_emotion": target_emotion, "plans": revised_plans},
            analysis,
            target_emotion,
        )
        value["verdict"] = str(value["verdict"]).lower()
        value["issues"] = issues
        value["revised_plans"] = validated["plans"]
        value["diversity_report"] = validated["diversity_report"]
        return value

    @classmethod
    def _validate_flux_prompt(
        cls, value: dict[str, Any], instruction: dict[str, Any]
    ) -> dict[str, Any]:
        require_keys(
            value,
            ("execution_decision", "decision_reason", "edit_prompt", "verification_spec"),
            "FLUX prompt",
        )
        decision = str(value["execution_decision"]).strip().casefold()
        if decision not in {"edit", "already_satisfied"}:
            raise ValueError("execution_decision must be edit or already_satisfied")
        value["execution_decision"] = decision
        if not str(value["decision_reason"]).strip():
            raise ValueError("decision_reason must cite current-image evidence")
        edit_prompt = str(value["edit_prompt"]).strip()
        if decision == "edit" and not edit_prompt.startswith("CONTROLLED REGION EDIT"):
            raise ValueError('edit_prompt must begin with "CONTROLLED REGION EDIT"')
        if decision == "already_satisfied" and edit_prompt:
            raise ValueError("already_satisfied requires an empty edit_prompt")
        spec = value["verification_spec"]
        if not isinstance(spec, dict):
            raise ValueError("verification_spec must be an object")
        require_keys(
            spec,
            (
                "required_change",
                "target_region",
                "placement_scale_depth",
                "must_remain_unchanged",
                "must_not_appear",
                "success_criteria",
            ),
            "verification_spec",
        )
        generic = re.compile(
            r"^(?:one objectively checkable visible result|preserve semantics|"
            r"make it more emotional|instruction specific visible end state)$",
            re.IGNORECASE,
        )
        for field in ("required_change", "target_region"):
            text = str(spec[field]).strip()
            if not text or generic.match(text):
                raise ValueError(f"verification_spec.{field} must be instruction-specific")
        for field in (
            "placement_scale_depth",
            "must_remain_unchanged",
            "must_not_appear",
            "success_criteria",
        ):
            items = spec[field]
            if not isinstance(items, list) or not items or any(not str(item).strip() for item in items):
                raise ValueError(f"verification_spec.{field} must be a non-empty string list")
        if len(spec["success_criteria"]) < 2:
            raise ValueError("verification_spec.success_criteria needs change and preservation checks")
        generic_target_tokens = {
            "existing", "scene", "background", "image", "region", "visual", "content",
            "lighting", "color", "treatment", "number", "appearance",
        }
        target_tokens = {
            token
            for token in cls._normalize_plan_text(instruction.get("operation_target")).split()
            if len(token) >= 4 and token not in generic_target_tokens
        }
        specification_text = cls._normalize_plan_text(
            f"{spec['required_change']} {spec['target_region']}"
        )
        if target_tokens and not (target_tokens & set(specification_text.split())):
            raise ValueError("verification_spec does not name the approved operation target")
        return value

    @classmethod
    def _validate_flux_decision(
        cls,
        value: dict[str, Any],
        instruction: dict[str, Any],
        require_edit: bool,
    ) -> dict[str, Any]:
        validated = cls._validate_flux_prompt(value, instruction)
        if require_edit and validated["execution_decision"] == "already_satisfied":
            # The three-image validator is authoritative about whether a no-op
            # was actually satisfied.  Do not spend JSON retries asking a
            # stubborn composer to reverse itself: deterministically convert
            # the already validated specification into a controlled edit.
            original_reason = str(validated.get("decision_reason", "")).strip()
            specification = validated["verification_spec"]
            validated = {
                **validated,
                "execution_decision": "edit",
                "decision_reason": (
                    "OVERRIDDEN AFTER VISUAL VALIDATION: the previous no-op was disproved; "
                    "execute the approved localized instruction. Composer claim was: "
                    + original_reason
                ),
                "edit_prompt": cls._controlled_edit_prompt(instruction, specification),
                "decision_override": {
                    "original_decision": "already_satisfied",
                    "reason": "three-image step validation disproved the requested visible state",
                    "policy": "validator-authoritative deterministic conversion to edit",
                },
            }
            validated = cls._validate_flux_prompt(validated, instruction)
        return validated

    @staticmethod
    def _controlled_edit_prompt(
        instruction: dict[str, Any], verification_spec: dict[str, Any]
    ) -> str:
        placement = "; ".join(
            str(item).strip()
            for item in verification_spec.get("placement_scale_depth", [])
            if str(item).strip()
        )
        preserve = "; ".join(
            str(item).strip()
            for item in verification_spec.get("must_remain_unchanged", [])
            if str(item).strip()
        )
        forbidden = "; ".join(
            str(item).strip()
            for item in verification_spec.get("must_not_appear", [])
            if str(item).strip()
        )
        return (
            "CONTROLLED REGION EDIT: Perform only this approved localized operation: "
            f"{str(instruction.get('instruction', '')).strip()} "
            f"Target region: {str(verification_spec.get('target_region', '')).strip()}. "
            f"Required visible end state: {str(verification_spec.get('required_change', '')).strip()}. "
            f"Placement, scale, and depth constraints: {placement}. "
            f"Keep unchanged: {preserve}. "
            f"Do not introduce: {forbidden}. "
            "Keep the main subject recognizable and the broad source scene coherent. Allow natural "
            "changes to nearby lighting, shadows, atmosphere, background details, pose, scale, and "
            "composition when they help realize the approved emotional edit."
        )

    @classmethod
    def _deterministic_edit_decision(
        cls,
        instruction: dict[str, Any],
        analysis: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        target = str(instruction.get("operation_target", "localized visible change")).strip()
        region = str(
            instruction.get("target_region", "the source-compatible region named by the instruction")
        ).strip()
        operation = str(instruction.get("instruction", "")).strip()
        preserve_items = [
            str(item).strip()
            for item in instruction.get("preserve", [])
            if str(item).strip()
        ]
        if not preserve_items:
            preserve_items = [
                "the main subject remains recognizable and the broad scene remains coherent"
            ]
        specification = {
            "required_change": f"The approved visible change to {target} is clearly present: {operation}",
            "target_region": region,
            "placement_scale_depth": [
                f"center the requested change around {region} while blending it naturally into nearby areas",
                "use a visually meaningful scale and allow supporting light, shadow, and atmosphere changes",
            ],
            "must_remain_unchanged": preserve_items,
            "must_not_appear": [
                "no replacement of the main subject, unrelated scene, illegible text, or severe visual artifact"
            ],
            "success_criteria": [
                f"the requested change to {target} is visibly present in {region}",
                "the main subject remains recognizable and the edited scene remains coherent",
            ],
        }
        value = {
            "execution_decision": "edit",
            "decision_reason": reason,
            "edit_prompt": cls._controlled_edit_prompt(instruction, specification),
            "verification_spec": specification,
            "decision_override": {
                "reason": reason,
                "policy": "deterministic fallback after composer schema/obedience failure",
            },
        }
        return cls._validate_flux_prompt(value, instruction)

    @staticmethod
    def _validate_step(
        value: dict[str, Any], execution_decision: str, verification_spec: dict[str, Any]
    ) -> dict[str, Any]:
        normalization_warnings: list[str] = []
        verdict = str(value.get("verdict", "")).lower()
        if verdict not in {"correct", "wrong"}:
            verdict = "wrong"
            normalization_warnings.append(
                "invalid or missing verdict was conservatively normalized to wrong"
            )
        value["verdict"] = verdict
        value.setdefault("problems", [])
        value.setdefault("retry_instruction", "")
        value.setdefault("facial_expression_constraint_obeyed", True)
        preservation_fields = (
            "required_state_satisfied",
            "preservation_ok",
            "emotion_direction_ok",
            "facial_expression_constraint_obeyed",
            "main_subject_preserved",
            "key_information_preserved",
            "composition_preserved",
            "new_content_context_compatible",
            "brightness_and_detail_preserved",
        )
        for field in preservation_fields:
            value.setdefault(field, False)
            if not isinstance(value[field], bool):
                value[field] = False
                normalization_warnings.append(f"non-boolean {field} was normalized to false")
        if not all(value[field] for field in preservation_fields):
            value["verdict"] = "wrong"
        value.setdefault("change_localized", False)
        if not isinstance(value["change_localized"], bool):
            value["change_localized"] = False
            normalization_warnings.append("non-boolean change_localized was normalized to false")
        instruction_executed = value.get("instruction_executed")
        if not isinstance(instruction_executed, bool):
            instruction_executed = False
            value["instruction_executed"] = False
            normalization_warnings.append(
                "missing/non-boolean instruction_executed was normalized to false"
            )
        if execution_decision == "edit" and (
            not instruction_executed or not value["change_localized"]
        ):
            value["verdict"] = "wrong"
        criteria = verification_spec["success_criteria"]
        results = value.get("criterion_results")
        supplied_results = results if isinstance(results, list) else []
        normalized_results: list[dict[str, Any]] = []
        for index, expected in enumerate(criteria):
            supplied = supplied_results[index] if index < len(supplied_results) else {}
            if not isinstance(supplied, dict):
                supplied = {}
            supplied_criterion = str(supplied.get("criterion", "")).strip()
            passed = supplied.get("passed")
            evidence = str(supplied.get("evidence", "")).strip()
            schema_ok = isinstance(passed, bool) and bool(evidence)
            normalized_results.append(
                {
                    "criterion": str(expected).strip(),
                    "passed": bool(passed) if schema_ok else False,
                    "evidence": evidence or "No valid visible evidence was supplied for this criterion.",
                    **(
                        {"reported_criterion": supplied_criterion}
                        if supplied_criterion and supplied_criterion != str(expected).strip()
                        else {}
                    ),
                }
            )
        if len(supplied_results) != len(criteria):
            normalization_warnings.append(
                "wrong number of criterion results; missing items failed conservatively"
            )
        value["criterion_results"] = normalized_results
        if not all(bool(result["passed"]) for result in normalized_results):
            value["verdict"] = "wrong"
        if normalization_warnings:
            value.setdefault("problems", []).extend(normalization_warnings)
            value["validation_normalization"] = {
                "policy": "schema drift is normalized conservatively instead of aborting the pipeline",
                "warnings": normalization_warnings,
            }
        return value

    @staticmethod
    def _validate_critic(value: dict[str, Any]) -> dict[str, Any]:
        normalization_warnings: list[str] = []
        verdict = str(value.get("verdict", "")).lower()
        if verdict not in {"correct", "wrong"}:
            verdict = "wrong"
            normalization_warnings.append("invalid or missing verdict was normalized to wrong")
        value["verdict"] = verdict
        value.setdefault("perceived_emotion", "undetermined because critic output was incomplete")
        value.setdefault("emotion_evidence", {})
        value.setdefault("semantic_preservation", "poor")
        value.setdefault("instruction_diagnosis", [])
        value.setdefault("updated_instructions", [])
        value.setdefault("final_reason", "Incomplete critic response; candidate rejected conservatively.")
        if not isinstance(value["updated_instructions"], list):
            value["updated_instructions"] = []
            normalization_warnings.append("updated_instructions was normalized to an empty list")
        preservation_fields = (
            "facial_expression_constraint_obeyed",
            "main_subject_preserved",
            "key_information_preserved",
            "composition_preserved",
            "new_content_context_compatible",
            "multiple_emotion_factors_present",
            "brightness_and_detail_preserved",
        )
        for field in preservation_fields:
            if not isinstance(value.get(field), bool):
                value[field] = False
                normalization_warnings.append(f"missing/non-boolean {field} was normalized to false")
        if not all(value[field] for field in preservation_fields):
            value["verdict"] = "wrong"
        if normalization_warnings:
            value["critic_normalization"] = {
                "policy": "schema drift is normalized conservatively instead of aborting other plans",
                "warnings": normalization_warnings,
            }
        return value

    @staticmethod
    def _objective_image_checks(
        reference_path: Path, edited_path: Path, target_emotion: str
    ) -> dict[str, Any]:
        with Image.open(reference_path) as image:
            reference = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(edited_path) as image:
            edited_image = image.convert("RGB")
            if edited_image.size != (reference.shape[1], reference.shape[0]):
                edited_image = edited_image.resize(
                    (reference.shape[1], reference.shape[0]), Image.Resampling.LANCZOS
                )
            edited = np.asarray(edited_image, dtype=np.float32) / 255.0

        coefficients = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        reference_luma = reference @ coefficients
        edited_luma = edited @ coefficients
        reference_mean = float(reference_luma.mean())
        edited_mean = float(edited_luma.mean())
        brightness_ratio = edited_mean / max(reference_mean, 1e-6)
        near_black_fraction = float((edited_luma < 0.05).mean())

        def edge_map(luma: np.ndarray) -> np.ndarray:
            horizontal = np.diff(luma, axis=1, append=luma[:, -1:])
            vertical = np.diff(luma, axis=0, append=luma[-1:, :])
            return np.sqrt(horizontal * horizontal + vertical * vertical)

        reference_edges = edge_map(reference_luma).reshape(-1)
        edited_edges = edge_map(edited_luma).reshape(-1)
        if float(reference_edges.std()) < 1e-8 or float(edited_edges.std()) < 1e-8:
            edge_correlation = 0.0
        else:
            edge_correlation = float(np.corrcoef(reference_edges, edited_edges)[0, 1])
        mean_absolute_pixel_change = float(np.abs(reference - edited).mean())

        # Keep only a corruption/illegibility guard here.  Emotional edits are
        # allowed to change exposure and large portions of the scene; semantic
        # preservation is judged by the vision critic at subject/scene level.
        minimum_ratio = 0.12 if target_emotion == "fear" else 0.10
        maximum_black = 0.75 if target_emotion == "fear" else 0.80
        brightness_ok = brightness_ratio >= minimum_ratio and near_black_fraction <= maximum_black
        structurally_identical = mean_absolute_pixel_change <= 1e-6
        structural_ok = structurally_identical or (
            edge_correlation >= -0.05 and mean_absolute_pixel_change <= 0.70
        )
        return {
            "reference_mean_luminance": reference_mean,
            "edited_mean_luminance": edited_mean,
            "brightness_ratio": brightness_ratio,
            "near_black_fraction": near_black_fraction,
            "minimum_brightness_ratio": minimum_ratio,
            "maximum_near_black_fraction": maximum_black,
            "brightness_ok": brightness_ok,
            "edge_correlation": edge_correlation,
            "minimum_edge_correlation": -0.05,
            "mean_absolute_pixel_change": mean_absolute_pixel_change,
            "maximum_mean_absolute_pixel_change": 0.70,
            "structural_ok": structural_ok,
            "objective_preservation_ok": brightness_ok and structural_ok,
        }

    def _predict(self, image_path: Path) -> dict[str, float]:
        if self.classifier is None:
            return empty_emotion_probabilities()
        return self.classifier.predict(image_path)

    def _execute_instruction(
        self,
        plan_dir: Path,
        plan_id: str,
        instruction: dict[str, Any],
        analysis: dict[str, Any],
        source_emotion_label: str,
        target_emotion: str,
        source_image: Path,
        input_image: Path,
        seed_base: int,
        phase: str,
        completed_steps: list[dict[str, Any]] | None = None,
        prior_step_failures: list[dict[str, Any]] | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        step_id = int(instruction["step_id"])
        self._assert_face_constraint(instruction, analysis)
        step_dir = plan_dir / phase / f"step_{step_id:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        copy_image(input_image, step_dir / "input.png")
        correction: str | None = None
        attempts: list[dict[str, Any]] = []
        last_output = input_image
        forced_edit_after_false_noop = False
        for attempt in range(1, self.config.max_step_retries + 2):
            system, user = flux_prompt_generation(
                analysis,
                source_emotion_label,
                target_emotion,
                instruction,
                completed_steps=completed_steps,
                correction=correction,
                prior_step_failures=prior_step_failures,
            )
            try:
                prompt_data = self._ask_json(
                    "compose_edit_prompt",
                    "Editing Agent",
                    system,
                    user,
                    image_paths=[input_image],
                    validator=lambda value: self._validate_flux_decision(
                        value,
                        instruction,
                        forced_edit_after_false_noop,
                    ),
                )
            except RuntimeError as exc:
                # Prompt composition is advisory; the approved plan instruction
                # and source constraints are sufficient to build a safe edit.
                # Keep malformed VLM JSON from aborting all remaining plans.
                prompt_data = self._deterministic_edit_decision(
                    instruction,
                    analysis,
                    "Editing-agent prompt composition exhausted schema retries: " + str(exc),
                )
                self._event(
                    "Editing Agent",
                    "compose_edit_prompt_fallback",
                    {
                        "plan_id": plan_id,
                        "step_id": step_id,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                    prompt_data,
                )
            prompt = str(prompt_data["edit_prompt"])
            output_path = step_dir / f"output_try_{attempt:02d}.png"
            seed = seed_base + step_id * 100 + attempt
            write_json(
                step_dir / f"prompt_try_{attempt:02d}.json",
                {**prompt_data, "combined_prompt": prompt, "seed": seed},
            )
            if prompt_data["execution_decision"] == "already_satisfied":
                copy_image(input_image, output_path)
                self._event(
                    "Editing Agent",
                    "skip_edit_already_satisfied",
                    {
                        "plan_id": plan_id,
                        "step_id": step_id,
                        "attempt": attempt,
                        "visible_reason": prompt_data["decision_reason"],
                    },
                    {"output": str(output_path), "pixel_edit_performed": False},
                )
            else:
                self.editor.edit(input_image, prompt, output_path, seed)
                self._event(
                    "Editing Agent",
                    "flux2_edit",
                    {
                        "plan_id": plan_id,
                        "step_id": step_id,
                        "attempt": attempt,
                        "prompt": prompt,
                        "seed": seed,
                    },
                    {"output": str(output_path)},
                )
            validation_system, validation_user = step_validation_prompt(
                source_emotion_label,
                target_emotion,
                instruction,
                prompt_data["verification_spec"],
                prompt_data["execution_decision"],
                bool(analysis.get("facial_expression_available")),
                analysis,
            )
            try:
                validation = self._ask_json(
                    "edit_reactify",
                    "Editing Agent",
                    validation_system,
                    validation_user,
                    image_paths=[source_image, input_image, output_path],
                    validator=lambda value: self._validate_step(
                        value,
                        prompt_data["execution_decision"],
                        prompt_data["verification_spec"],
                    ),
                )
            except RuntimeError as exc:
                criteria = prompt_data["verification_spec"]["success_criteria"]
                validation = self._validate_step(
                    {
                        "verdict": "wrong",
                        "required_state_satisfied": False,
                        "instruction_executed": prompt_data["execution_decision"] == "edit",
                        "change_localized": False,
                        "preservation_ok": False,
                        "emotion_direction_ok": False,
                        "facial_expression_constraint_obeyed": True,
                        "main_subject_preserved": False,
                        "key_information_preserved": False,
                        "composition_preserved": False,
                        "new_content_context_compatible": False,
                        "brightness_and_detail_preserved": False,
                        "criterion_results": [
                            {
                                "criterion": criterion,
                                "passed": False,
                                "evidence": "No schema-valid visual-validation evidence was returned.",
                            }
                            for criterion in criteria
                        ],
                        "problems": [
                            "Step visual validation exhausted schema retries; reject and roll back "
                            "this candidate edit."
                        ],
                        "retry_instruction": (
                            "Retry the same approved localized operation conservatively and preserve "
                            "all named invariants."
                        ),
                        "validation_fallback": {"error": str(exc)},
                    },
                    prompt_data["execution_decision"],
                    prompt_data["verification_spec"],
                )
                self._event(
                    "Editing Agent",
                    "edit_reactify_fallback",
                    {
                        "plan_id": plan_id,
                        "step_id": step_id,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                    validation,
                )
            objective_checks = self._objective_image_checks(
                input_image, output_path, target_emotion
            )
            validation["objective_image_checks"] = objective_checks
            if not objective_checks["objective_preservation_ok"]:
                validation["verdict"] = "wrong"
                if not objective_checks["brightness_ok"]:
                    validation["brightness_and_detail_preserved"] = False
                    validation.setdefault("problems", []).append(
                        "Objective luminance gate failed: preserve normal exposure and readable midtone detail."
                    )
                if not objective_checks["structural_ok"]:
                    validation["composition_preserved"] = False
                    validation.setdefault("problems", []).append(
                        "Objective structure/change gate failed: keep source geometry and reduce edit magnitude."
                    )
                validation["retry_instruction"] = (
                    "Redo the emotional edit while keeping the main subject recognizable and the image readable. "
                    "Avoid a fully black, corrupted, or unrelated result; substantial coherent scene changes are allowed."
                )
            attempt_record = {
                "attempt": attempt,
                "seed": seed,
                "input_image": str(input_image),
                "output_image": str(output_path),
                "prompt": prompt_data,
                "validation": validation,
            }
            attempts.append(attempt_record)
            write_json(step_dir / f"validation_try_{attempt:02d}.json", validation)
            last_output = output_path
            if validation["verdict"] == "correct":
                break
            problems = validation.get("problems") or []
            correction = str(validation.get("retry_instruction") or "; ".join(map(str, problems)))
            if prompt_data["execution_decision"] == "already_satisfied":
                forced_edit_after_false_noop = True
                correction = (
                    "The validator found that the requested visible state is not actually satisfied. "
                    "Set execution_decision to edit and perform the approved localized operation. "
                    + correction
                ).strip()
        step_accepted = attempts[-1]["validation"]["verdict"] == "correct"
        if not step_accepted:
            # A rejected FLUX result must not become the input to the next step.
            # Roll back to the last verified image so errors cannot accumulate.
            last_output = input_image
        result = {
            "plan_id": plan_id,
            "phase": phase,
            "step_id": step_id,
            "instruction": instruction,
            "attempts": attempts,
            "accepted": step_accepted,
            "rollback_applied": not step_accepted,
            "execution_decision": attempts[-1]["prompt"]["execution_decision"],
            "prior_step_failures_received": prior_step_failures or [],
            "failure_feedback_for_next_step": (
                None
                if step_accepted
                else {
                    "failed_step_id": step_id,
                    "failed_instruction": instruction.get("instruction"),
                    "problems": attempts[-1]["validation"].get("problems", []),
                    "retry_instruction": attempts[-1]["validation"].get(
                        "retry_instruction", ""
                    ),
                    "objective_image_checks": attempts[-1]["validation"].get(
                        "objective_image_checks", {}
                    ),
                    "action_taken": "rejected output rolled back to last verified input",
                }
            ),
            "selected_output": str(last_output),
        }
        write_json(step_dir / "step_result.json", result)
        return last_output, result

    def _run_plan(
        self,
        run_dir: Path,
        source_image: Path,
        analysis: dict[str, Any],
        source_emotion_label: str,
        target_emotion: str,
        plan: dict[str, Any],
        plan_index: int,
        source_probabilities: dict[str, float],
    ) -> dict[str, Any]:
        plan_id = str(plan["plan_id"])
        plan_dir = run_dir / plan_id
        plan_dir.mkdir(parents=True, exist_ok=True)
        write_json(plan_dir / "plan.json", plan)
        current = source_image
        step_results: list[dict[str, Any]] = []
        completed_steps: list[dict[str, Any]] = []
        prior_step_failures: list[dict[str, Any]] = []
        seed_base = self.config.seed + plan_index * 100_000
        for instruction in plan["instructions"]:
            current, result = self._execute_instruction(
                plan_dir,
                plan_id,
                instruction,
                analysis,
                source_emotion_label,
                target_emotion,
                source_image,
                current,
                seed_base,
                phase="pre_creation",
                completed_steps=completed_steps,
                prior_step_failures=prior_step_failures,
            )
            step_results.append(result)
            if result["accepted"]:
                completed_steps.append(
                    {
                        "step_id": result["step_id"],
                        "instruction": result["instruction"].get("instruction"),
                        "execution_decision": result["execution_decision"],
                        "verified_required_change": result["attempts"][-1]["prompt"]
                        ["verification_spec"]["required_change"],
                    }
                )
            if result["failure_feedback_for_next_step"]:
                prior_step_failures.append(result["failure_feedback_for_next_step"])

        status = "unresolved"
        critic_history: list[dict[str, Any]] = []
        for critic_round in range(self.config.max_critic_rounds + 1):
            edited_probabilities = self._predict(current)
            system, user = critic_prompt(
                source_emotion_label,
                target_emotion,
                analysis,
                plan,
                step_results,
                source_probabilities,
                edited_probabilities,
            )
            try:
                feedback = self._ask_json(
                    "evaluate_edit_emotion",
                    "Critic Agent",
                    system,
                    user,
                    image_paths=[source_image, current],
                    validator=self._validate_critic,
                    max_new_tokens=1536,
                )
            except RuntimeError as exc:
                feedback = self._validate_critic(
                    {
                        "verdict": "wrong",
                        "perceived_emotion": "undetermined because critic schema retries failed",
                        "emotion_evidence": {},
                        "semantic_preservation": "poor",
                        "facial_expression_constraint_obeyed": False,
                        "main_subject_preserved": False,
                        "key_information_preserved": False,
                        "composition_preserved": False,
                        "new_content_context_compatible": False,
                        "multiple_emotion_factors_present": False,
                        "brightness_and_detail_preserved": False,
                        "instruction_diagnosis": [],
                        "updated_instructions": [],
                        "final_reason": (
                            "Critic exhausted schema retries; candidate rejected conservatively "
                            "without aborting the other plans."
                        ),
                        "critic_fallback": {"error": str(exc)},
                    }
                )
                self._event(
                    "Critic Agent",
                    "evaluate_edit_emotion_fallback",
                    {"plan_id": plan_id, "round": critic_round, "error": str(exc)},
                    feedback,
                )
            objective_checks = self._objective_image_checks(
                source_image, current, target_emotion
            )
            feedback["objective_image_checks"] = objective_checks
            if not objective_checks["objective_preservation_ok"]:
                feedback["verdict"] = "wrong"
                if not objective_checks["brightness_ok"]:
                    feedback["brightness_and_detail_preserved"] = False
                if not objective_checks["structural_ok"]:
                    feedback["composition_preserved"] = False
                feedback.setdefault("updated_instructions", []).append(
                    "Redo the intended emotional transformation while keeping the main subject recognizable and "
                    "the broad scene coherent. Avoid a corrupted, nearly black, or unrelated result; retain a clear "
                    "emotion-bearing semantic or atmospheric change."
                )
            feedback["round"] = critic_round
            feedback["edited_probabilities"] = edited_probabilities
            feedback["image"] = str(current)
            write_json(plan_dir / f"critic_feedback_round_{critic_round:02d}.json", feedback)
            critic_history.append(feedback)
            resnet_ok = (
                not self.config.require_resnet_target
                or max(edited_probabilities, key=edited_probabilities.get) == target_emotion
            )
            if (
                feedback["verdict"] == "correct"
                and resnet_ok
            ):
                status = "accepted"
                break
            if critic_round >= self.config.max_critic_rounds:
                break
            updates = [str(item).strip() for item in feedback["updated_instructions"] if str(item).strip()]
            if not updates:
                break
            for update_index, update in enumerate(updates, 1):
                revised_instruction = {
                    "step_id": update_index,
                    "element_type": "critic_revision",
                    "method": "attribute_change",
                    "operation_target": update,
                    "target_region": "the specific region named by the critic correction",
                    "instruction": update,
                    "emotion_rationale": f"Critic revision for {target_emotion}",
                    "preserve": ["main subject", "identity", "layout", "unrelated content"],
                }
                revised_instruction = self._normalize_instruction_constraints(
                    revised_instruction,
                    analysis,
                    target_emotion,
                    update_index,
                )
                current, result = self._execute_instruction(
                    plan_dir,
                    plan_id,
                    revised_instruction,
                    analysis,
                    source_emotion_label,
                    target_emotion,
                    source_image,
                    current,
                    seed_base + (critic_round + 1) * 10_000,
                    phase=f"optimization_round_{critic_round + 1:02d}",
                    completed_steps=completed_steps,
                    prior_step_failures=prior_step_failures,
                )
                step_results.append(result)
                if result["accepted"]:
                    completed_steps.append(
                        {
                            "step_id": result["step_id"],
                            "instruction": result["instruction"].get("instruction"),
                            "execution_decision": result["execution_decision"],
                            "verified_required_change": result["attempts"][-1]["prompt"]
                            ["verification_spec"]["required_change"],
                        }
                    )
                if result["failure_feedback_for_next_step"]:
                    prior_step_failures.append(result["failure_feedback_for_next_step"])

        final_path = plan_dir / "final.png"
        copy_image(current, final_path)
        result = {
            "plan_id": plan_id,
            "status": status,
            "final_image": str(final_path),
            "step_results": step_results,
            "critic_history": critic_history,
            "preservation_gate_passed": bool(
                critic_history
                and critic_history[-1].get("main_subject_preserved")
                and critic_history[-1].get("key_information_preserved")
                and critic_history[-1].get("composition_preserved")
                and critic_history[-1].get("new_content_context_compatible")
                and critic_history[-1].get("multiple_emotion_factors_present")
                and critic_history[-1].get("brightness_and_detail_preserved")
                and critic_history[-1].get("objective_image_checks", {}).get(
                    "objective_preservation_ok"
                )
            ),
            "final_probabilities": self._predict(final_path),
        }
        write_json(plan_dir / "result.json", result)
        return result

    def run(
        self,
        source: Path,
        target_emotion: str,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        if target_emotion not in EMOTIONS:
            raise ValueError(f"target emotion must be one of {EMOTIONS}")
        run_name = run_name or f"{source.stem}_{target_emotion}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.config.output_root / run_name
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.trajectory_path = run_dir / "trajectory.jsonl"
        source_image = run_dir / "source.png"
        image_info = prepare_source_image(source, source_image, self.config.image_size)
        write_json(
            run_dir / "run_manifest.json",
            {
                "target_emotion": target_emotion,
                "config": {**asdict(self.config), "output_root": str(self.config.output_root)},
                "image": image_info,
                "paper_workflow": ["pre_creation", "optimization"],
                "editing_backend_deviation": "Paper multi-tool library replaced by task-specified FLUX.2-klein-9B",
                "source_emotion_policy": (
                    "The inference dataset is unlabeled, so the Planning Agent predicts source emotion "
                    "from visible evidence as described by the paper."
                ),
            },
        )

        analysis_system, analysis_user = source_analysis_prompt()
        analysis = self._ask_json(
            "input_query",
            "Planning Agent",
            analysis_system,
            analysis_user,
            image_paths=[source_image],
            validator=self._validate_analysis,
        )
        source_emotion_label = str(analysis["source_emotion"])
        source_emotion_provenance = "planning_agent_vlm_prediction"
        analysis["source_emotion_provenance"] = source_emotion_provenance
        write_json(
            run_dir / "run_manifest.json",
            {
                **{
                    "target_emotion": target_emotion,
                    "config": {**asdict(self.config), "output_root": str(self.config.output_root)},
                    "image": image_info,
                    "paper_workflow": ["pre_creation", "optimization"],
                    "editing_backend_deviation": "Paper multi-tool library replaced by task-specified FLUX.2-klein-9B",
                    "source_emotion_policy": (
                        "The inference dataset is unlabeled, so the Planning Agent predicts source emotion "
                        "from visible evidence as described by the paper."
                    ),
                },
                "source_emotion": source_emotion_label,
                "source_emotion_provenance": source_emotion_provenance,
            },
        )
        write_json(run_dir / "source_analysis.json", analysis)

        available_element_types = [
            element_type
            for element_type in ELEMENT_TYPES
            if analysis["facial_expression_available"] or element_type != "facial_expression"
        ]
        neutral_query = retrieval_query(analysis["retrieval_semantics"])
        retrieved = self.retriever.retrieve_diverse(
            neutral_query,
            target_emotion,
            self.config.top_k,
            allowed_element_types=available_element_types,
        )
        retrieval_record = {
            "embedding_query": neutral_query,
            "query_policy": (
                "Embed affect-neutral visible source semantics once; apply target emotion and element "
                "type as metadata filters; select a globally ranked diverse top-k pool."
            ),
            "target_emotion": target_emotion,
            "facial_expression_available": analysis["facial_expression_available"],
            "excluded_elements": (
                [] if analysis["facial_expression_available"] else ["facial_expression"]
            ),
            "results": retrieved,
        }
        write_json(run_dir / "efk_retrieval.json", retrieval_record)
        self._event(
            "Planning Agent",
            "RAG_query",
            {
                "embedding_query": neutral_query,
                "emotion_filter": target_emotion,
                "allowed_element_types": available_element_types,
            },
            retrieved,
        )

        plan_system, plan_user = plan_generation_prompt(
            analysis,
            source_emotion_label,
            target_emotion,
            retrieved,
            self.config.num_plans,
            self.config.steps_per_plan,
        )
        plan_bundle = self._ask_json(
            "generate_diverse_plans",
            "Planning Agent",
            plan_system,
            plan_user,
            image_paths=[source_image],
            validator=lambda value: self._validate_plans(
                value, analysis, target_emotion
            ),
            max_new_tokens=3072,
        )

        system, user = planning_evaluation_prompt(
            analysis,
            source_emotion_label,
            target_emotion,
            plan_bundle["plans"],
            self.config.steps_per_plan,
            plan_bundle["diversity_report"],
        )
        try:
            evaluation = self._ask_json(
                "Planning_evaluate",
                "Planning Agent",
                system,
                user,
                image_paths=[source_image],
                validator=lambda value: self._validate_plan_evaluation(
                    value, analysis, target_emotion
                ),
                max_new_tokens=4096,
            )
            evaluated_plans = evaluation["revised_plans"]
        except RuntimeError as exc:
            # Generation already passed full bundle validation.  Preserve that
            # valid bundle if the optional joint evaluator cannot return schema.
            evaluated_plans = plan_bundle["plans"]
            evaluation = {
                "verdict": "reasonable",
                "issues": ["Evaluator schema retries exhausted; retained validated bundle."],
                "revised_plans": evaluated_plans,
            }
            self._event(
                "Planning Agent",
                "Planning_evaluate_fallback",
                {"error": str(exc)},
                evaluation,
            )
        # Mandatory second global check after evaluator revisions and face repairs.
        final_plan_bundle = self._validate_plans(
            {"target_emotion": target_emotion, "plans": evaluated_plans},
            analysis,
            target_emotion,
        )
        evaluated_plans = final_plan_bundle["plans"]
        write_json(
            run_dir / "plans.json",
            {
                "target_emotion": target_emotion,
                "joint_evaluation": evaluation,
                "diversity_report": final_plan_bundle["diversity_report"],
                "plans": evaluated_plans,
            },
        )

        source_probabilities = self._predict(source_image)
        write_json(run_dir / "source_emotion_probabilities.json", source_probabilities)
        results = [
            self._run_plan(
                run_dir,
                source_image,
                analysis,
                source_emotion_label,
                target_emotion,
                plan,
                plan_index,
                source_probabilities,
            )
            for plan_index, plan in enumerate(evaluated_plans, 1)
        ]
        summary = {
            "run_dir": str(run_dir),
            "source_image": str(source_image),
            "source_probabilities": source_probabilities,
            "source_emotion": source_emotion_label,
            "source_emotion_provenance": source_emotion_provenance,
            "target_emotion": target_emotion,
            "accepted": sum(result["status"] == "accepted" for result in results),
            "total": len(results),
            "results": [
                {
                    "plan_id": result["plan_id"],
                    "status": result["status"],
                    "final_image": result["final_image"],
                    "preservation_gate_passed": result["preservation_gate_passed"],
                    "final_probabilities": result["final_probabilities"],
                }
                for result in results
            ],
        }
        write_json(run_dir / "summary.json", summary)
        return summary
