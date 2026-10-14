from __future__ import annotations

import json
from typing import Any

from .constants import EDIT_METHODS, ELEMENT_TYPES


# These strings retain the analysis wording visible in the paper. The runtime
# asks the VLM for source emotion because the inference dataset is unlabeled.
PAPER_SOURCE_ANALYSIS = (
    "Analyze the semantic of this image, including color tone, background, object, "
    "subject attributes, human or animal expressions, etc."
)
PAPER_SOURCE_QUERY = "What are the semantics and emotion of the original image?"
RUNTIME_SOURCE_QUERY = "What are the visible semantics and source emotion of the original image?"


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def source_analysis_prompt() -> tuple[str, str]:
    system = """You are the Planning Agent's source-image analyst in EmoAgent. Analyze visible evidence and classify the source emotion into exactly one allowed category. Identify only the main subject and the broad scene structure that make the image recognizable; do not freeze every object or incidental detail. Keep semantic_summary and retrieval_semantics affect-neutral. Do not propose edits. Return valid JSON only, without Markdown."""
    user = f"""{RUNTIME_SOURCE_QUERY}

Use the visual-analysis scope shown in the paper:
{PAPER_SOURCE_ANALYSIS}

The inference dataset has no source-emotion labels. Infer source_emotion from visible evidence only and select exactly one of: amusement, awe, contentment, excitement, anger, disgust, fear, sadness, or mixed/uncertain. The later target emotion is deliberately not shown to you and must not bias this judgment. Explain the prediction briefly in source_emotion_evidence. Never insert the predicted label or affective interpretations such as excited, joyful, sad, frightening, amusing, anticipation, or contentment into semantic_summary or retrieval_semantics.

Use protected_entities sparingly: include only the primary identity-bearing subject or subjects, not every visible object. Protect recognizability and identity, while treating pose, expression, clothing details, surface appearance, lighting, color, surrounding objects, atmosphere, and background as editable whenever a later instruction requests them. Keep must_preserve_layout high-level rather than pixel-exact. Do not place the same property in both immutable_attributes and editable_attributes.

Return exactly this JSON structure:
{{
  "main_subject": "concise visible description without emotion words",
  "objects": ["visible object with approximate count when useful"],
  "scene": "visible setting and background without emotional interpretation",
  "composition": "camera viewpoint, framing, depth layers, layout, and spatial relations",
  "object_attributes": ["directly visible physical attribute"],
  "expressions_actions": ["directly visible expression or action; do not infer feelings"],
  "visible_people_or_faces": ["only clearly visible people, faces, or animals with readable faces"],
  "facial_expression_available": false,
  "protected_entities": [
    {{
      "entity": "salient subject or object",
      "approximate_count": "visible count or count range",
      "preserve_identity": ["identity-defining visible feature"],
      "preserve_geometry": ["only broad geometry needed to keep the main subject recognizable"],
      "immutable_attributes": ["only identity-defining attribute that truly must not change"],
      "editable_attributes": ["pose, expression, surface, color, lighting, surrounding context, or other non-identity attribute"]
    }}
  ],
  "must_preserve_layout": ["broad scene relationship needed for recognizability"],
  "editable_or_replaceable_elements": ["background, atmosphere, non-key object, or non-identity subject attribute that may change substantially"],
  "key_information_rationale": "why the protected entities and layout define the image identity",
  "lighting": "directly visible illumination and exposure",
  "global_hue": "directly visible dominant colors",
  "source_emotion": "one of amusement, awe, contentment, excitement, anger, disgust, fear, sadness, or mixed/uncertain",
  "source_emotion_evidence": ["concise directly visible cue supporting the source-emotion prediction"],
  "semantic_summary": "one concise affect-neutral sentence describing visible content and layout",
  "retrieval_semantics": "a concrete affect-neutral retrieval description covering subject, setting, existing objects, spatial layout, lighting, and available regions"
}}

Set facial_expression_available to true only when the source visibly contains a face whose expression can be edited without inventing a person, face, or identity. Set it to false for landscapes, objects, distant or occluded people, unreadable faces, and uncertainty. Do not infer an invisible face."""
    return system, user


def retrieval_query(retrieval_semantics: str) -> str:
    """Return only neutral source semantics for Eq. (4) embedding retrieval.

    Target emotion and element type are metadata filters in the retriever, not words
    appended to this embedding query.
    """

    return " ".join(str(retrieval_semantics).split())


def plan_generation_prompt(
    analysis: dict[str, Any],
    source_emotion_label: str,
    target_emotion: str,
    retrieved: list[dict[str, Any]],
    num_plans: int,
    steps_per_plan: int,
) -> tuple[str, str]:
    facial_expression_available = bool(analysis.get("facial_expression_available"))
    allowed_elements = [
        element
        for element in ELEMENT_TYPES
        if facial_expression_available or element != "facial_expression"
    ]
    allowed_methods = list(EDIT_METHODS)
    if not facial_expression_available:
        allowed_methods.remove("expression_change")
    exclusive_cues = 1
    system = """You are the Planning Agent in EmoAgent. Translate a target emotion into context-aware, clearly visible editing strategies using the paper's three-layer decision space: target emotion -> editing element -> editing method. Preserve the recognizable main subject and broad scene coherence, while allowing substantial changes to emotional content. Return valid JSON only, without Markdown."""
    user = f"""User text prompt: make it {target_emotion}

Planning-Agent source-emotion prediction: {source_emotion_label}
Use it only to understand the requested source-to-target transition. Do not rewrite visible source semantics to fit either emotion label.

Source-image analysis:
{_dump(analysis)}

The EFK candidates below were retrieved by embedding only affect-neutral source semantics, filtering nodes by target emotion, and diversifying the top results across visual factors:
{_dump(retrieved)}

Generate exactly {num_plans} semantically distinct editing plans. Each plan must contain exactly {steps_per_plan} ordered instructions and represent an independent visual interpretation of {target_emotion}, not a paraphrase, permutation, or different random seed of another plan.

Bundle-level diversity rules:
- Give every plan one concrete strategy_anchor and at least {exclusive_cues} distinctive_cues that are not reused as the core cues of another plan.
- Shared source-compatible support operations are allowed when useful, but every pair must differ by at least one meaningful concrete operation. Two plans must never contain the same complete order-insensitive operation set.
- Changing only the order, side, size, wording, or random seed of the same operations does not create a new plan.
- Do not force every plan to use every retrieved candidate. Prefer different compatible subsets and different causal visual narratives.
- A retrieved factor already present in the image is useful only if the plan specifies a safe, visible attribute/action change; simply naming the existing subject is not an edit.
- Target-specific evidence matters: do not substitute generic arousal, crowd size, brightness, or saturation for the requested emotion in every plan.

Allowed editing elements: {', '.join(allowed_elements)}, object_attribute, background.
Allowed editing methods: {', '.join(allowed_methods)}.

Apply the taxonomy consistently:
- Adding a new entity or increasing its count uses method add. Its element_type describes the cue (usually object or action), not the depth location where it is placed.
- background_change is only for replacing or transforming an existing scene/background, not for adding one object in the background.
- attribute_change modifies a named property of an existing entity.
- filter_change modifies a global or regional color/lighting treatment.
- expression_change requires an actually visible editable face.

Each instruction must perform one concrete and clearly visible operation, name its operation_target and target_region, and contribute meaningfully to {target_emotion}. Preserve the identity of the main subject and enough of the broad scene to recognize the source, but allow noticeable changes to pose, expression, attributes, supporting objects, background, weather, lighting, and color. Non-key emotional cues may be added, removed, or replaced. The preserve list should contain only essential invariants, not every unchanged pixel or incidental object. Do not replace the entire image and do not rely on a color change alone.

This reproduction uses one global FLUX image editor instead of the paper's specialized tool library. Ask for decisive edits that are strong enough to change the perceived emotion. Added or replaced content should be visually coherent and correctly placed; it may become prominent when that is needed to communicate the target emotion, as long as the main subject remains recognizable and the image remains plausible. Do not directly spell the emotion label in the image. A single strong emotion cue or several complementary cues are both valid.

Facial-expression availability is {facial_expression_available}. If false, do not edit or invent a readable facial expression. Secondary contextual people may be added only when compatible, but they cannot be the sole or dominant evidence.

Return exactly:
{{
  "target_emotion": "{target_emotion}",
  "plans": [
    {{
      "plan_id": "plan_01",
      "strategy_summary": "concise description of this complete path",
      "strategy_anchor": "one concrete visual narrative unique within the bundle",
      "distinctive_cues": ["visible cue unique to this plan", "second unique cue when required"],
      "instructions": [
        {{
          "step_id": 1,
          "element_type": "one allowed element",
          "method": "one allowed method",
          "operation_target": "specific existing or new visual entity/property",
          "target_region": "specific image region and spatial relation",
          "instruction": "one imperative visible editing operation",
          "emotion_rationale": "why this visible change specifically supports {target_emotion}",
          "preserve": ["specific entity identity, geometry, attribute, or unaffected region that must remain unchanged"]
        }}
      ]
    }}
  ]
}}"""
    return system, user


def planning_evaluation_prompt(
    analysis: dict[str, Any],
    source_emotion_label: str,
    target_emotion: str,
    plans: list[dict[str, Any]],
    steps_per_plan: int,
    diversity_report: dict[str, Any] | None = None,
) -> tuple[str, str]:
    system = """You are Planning_evaluate in EmoAgent. Evaluate the complete plan bundle jointly before execution. Repair semantic duplicates, taxonomy errors, preservation conflicts, and source-incompatible operations. Return valid JSON only, without Markdown."""
    user = f"""Planning-Agent source-emotion prediction: {source_emotion_label}
Target emotion: {target_emotion}
Source analysis: {_dump(analysis)}
Complete candidate plan bundle: {_dump(plans)}
Deterministic preflight report (warnings require review/revision but did not abort generation): {_dump(diversity_report or {})}

Jointly check:
1. every instruction specifically and visibly contributes to the target emotion;
2. every operation is compatible with visible source semantics and uses EFK factors as inspiration rather than mandatory props;
3. the main subject remains recognizable and the broad scene remains coherent, while emotional content and non-identity attributes may change substantially;
4. element_type, method, operation_target, target_region, and instruction agree with one another;
5. instructions within a plan do not conflict or make a later step redundant;
6. plans are independent visual paths, not paraphrases, permutations, side swaps, or mostly shared operation sets;
7. each plan has at least one genuinely distinctive visible cue and every pair differs by at least one meaningful concrete operation; shared source-compatible support operations are allowed, but complete order-insensitive duplicate sets are not;
8. the complete plan contains visible semantic or atmospheric evidence for the target emotion rather than a negligible adjustment;
9. new content is relevant, plausible, and correctly located.

Resolve every preflight warning when feasible. Favor edits that visibly change the emotional reading of the image. Do not weaken a useful edit merely to keep incidental objects, exact pixel geometry, or the original atmosphere unchanged. If any plan duplicates another, revise the duplicate plan's concrete operations. Preserve any deterministic taxonomy repair. If facial_expression_available is false, do not invent a readable close-up face.

Return:
{{
  "verdict": "reasonable or revise",
  "issues": ["plan ids and concise actionable issue"],
  "revised_plans": ["all complete plans, including unchanged plans"]
}}

Always return the full set of plans. When reasonable, repeat the complete bundle unchanged. When revise, return a complete corrected bundle."""
    return system, user


def flux_prompt_generation(
    analysis: dict[str, Any],
    source_emotion_label: str,
    target_emotion: str,
    instruction: dict[str, Any],
    completed_steps: list[dict[str, Any]] | None = None,
    correction: str | None = None,
    prior_step_failures: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    system = """You are the Editing Agent in EmoAgent. Inspect the current image and convert one approved instruction into a clear, effective FLUX.2 edit. Preserve source recognizability without making the edit timid. Build instruction-specific verification criteria that are concise and permissive of natural supporting changes. Return valid JSON only, without Markdown."""
    correction_text = f"\nCurrent-step validation problem to correct: {correction}" if correction else ""
    prior_failure_text = (
        f"\nEarlier rejected edits were rolled back. Do not repeat these failures:\n{_dump(prior_step_failures)}"
        if prior_step_failures
        else ""
    )
    user = f"""The attached image is the current verified input state for this step.

Source analysis: {_dump(analysis)}
Planning-Agent source-emotion prediction: {source_emotion_label}
Target emotion: {target_emotion}
Current approved instruction: {_dump(instruction)}
Previously accepted steps in this plan: {_dump(completed_steps or [])}{correction_text}{prior_failure_text}

First inspect the current image. If the exact requested visible state is already clearly present—whether it existed in the source or was produced by an earlier accepted step—set execution_decision to already_satisfied and do not request another edit. Similar content, a planned-but-unexecuted step, or a different object/region is not sufficient. Otherwise set execution_decision to edit.

For edit:
- Request the current instruction as one clearly visible operation and begin edit_prompt with "CONTROLLED REGION EDIT" (this phrase is only a routing tag; it does not mean the change must be subtle).
- State the target region and the intended visible result. Give FLUX enough freedom to render the change naturally across nearby pixels, lighting, shadows, reflections, and atmosphere.
- Preserve the recognizable identity of the main subject and the broad scene type. Exact pose, silhouette, object count, local geometry, lighting, background detail, and composition may change when needed by the approved instruction.
- New emotional content may be prominent. It must look coherent rather than artificially pasted in.

Create verification_spec from this instruction. Make required_change specific and judge preservation at the level of recognizable subject and coherent scene, not pixel-exact correspondence. Keep must_remain_unchanged and must_not_appear short: prohibit only identity loss, severe artifacts, unrelated text, and scene collapse. Success requires a noticeable emotional edit, not minimal compliance.

For already_satisfied, edit_prompt must be empty, decision_reason must cite visible current-image evidence, and verification_spec must still define how to verify that the exact requested state is present and source invariants remain intact.

Facial-expression availability is {analysis.get('facial_expression_available', False)}. If false, do not request or validate a readable expression. For fear, preserve visible midtones and readable subject detail.

Return exactly:
{{
  "execution_decision": "edit or already_satisfied",
  "decision_reason": "specific visible evidence from the current image",
  "edit_prompt": "one self-contained imperative FLUX.2 prompt, or empty only for already_satisfied",
  "verification_spec": {{
    "required_change": "instruction-specific visible end state",
    "target_region": "instruction-specific image region and spatial relation",
    "placement_scale_depth": ["checkable placement, relative scale, depth, count, or orientation requirement"],
    "must_remain_unchanged": ["specific protected entity identity, geometry, attribute, or unaffected region"],
    "must_not_appear": ["specific unrequested edit, duplication, obstruction, replacement, or artifact"],
    "success_criteria": ["individually checkable required change", "individually checkable preservation condition"]
  }}
}}"""
    return system, user


def step_validation_prompt(
    source_emotion_label: str,
    target_emotion: str,
    instruction: dict[str, Any],
    verification_spec: dict[str, Any],
    execution_decision: str,
    facial_expression_available: bool,
    analysis: dict[str, Any],
) -> tuple[str, str]:
    system = """You are edit_reactify, the step-level self-critic in the EmoAgent Editing Agent. Three images are attached: original source, image before this step, and proposed result. Accept clear emotion-supporting edits when the main subject remains recognizable and the scene stays coherent. Return valid JSON only, without Markdown."""
    user = f"""Planning-Agent source-emotion prediction: {source_emotion_label}
Target emotion: {target_emotion}
Execution decision: {execution_decision}
Approved instruction: {_dump(instruction)}
Instruction-specific verification specification: {_dump(verification_spec)}
Protected entities: {_dump(analysis.get('protected_entities', []))}
Protected layout: {_dump(analysis.get('must_preserve_layout', []))}
Approved editable content: {_dump(analysis.get('editable_or_replaceable_elements', []))}

Use image 2 -> image 3 to check whether the requested change is clearly present. Use image 1 -> image 3 only to confirm that the main subject is still recognizable and the broad scene has not collapsed into an unrelated image. Do not reject useful edits for changes in exact position, scale, silhouette, lighting, background details, or incidental objects when those changes support the instruction.

If execution_decision is edit, required_change must be newly and clearly present in image 3 at the specified region. If execution_decision is already_satisfied, verify that the exact required state was already present in image 2 and remains present in image 3; no new pixel-level edit is required.
Set instruction_executed to false for already_satisfied because no new editor operation was run; this is valid only when required_state_satisfied is true and all preservation criteria pass. Set change_localized to true only for a newly executed localized edit; it may be false for already_satisfied.

Check every success_criteria item separately and provide concise visible evidence. Repeat each supplied criterion verbatim in criterion_results. A step passes when the requested semantic change is visible, supports the target emotion, preserves main-subject recognizability, and has no severe artifact. Natural secondary changes and broader atmospheric changes are acceptable. Reject only a missing/wrong edit, lost or replaced main subject, unrelated scene regeneration, conspicuous artifact, or illegible result.

Facial-expression availability is {facial_expression_available}. If false, reject invented or expression-edited readable faces. For fear, reject loss of readable midtones or subject detail.

Return:
{{
  "verdict": "correct or wrong",
  "required_state_satisfied": true,
  "instruction_executed": true,
  "change_localized": true,
  "preservation_ok": true,
  "emotion_direction_ok": true,
  "facial_expression_constraint_obeyed": true,
  "main_subject_preserved": true,
  "key_information_preserved": true,
  "composition_preserved": true,
  "new_content_context_compatible": true,
  "brightness_and_detail_preserved": true,
  "criterion_results": [
    {{"criterion": "one supplied success criterion", "passed": true, "evidence": "specific visible evidence"}}
  ],
  "problems": ["visible actionable failure"],
  "retry_instruction": "corrected imperative operation, or empty when correct"
}}"""
    return system, user


def critic_prompt(
    source_emotion_label: str,
    target_emotion: str,
    analysis: dict[str, Any],
    plan: dict[str, Any],
    step_results: list[dict[str, Any]],
    source_probabilities: dict[str, float],
    edited_probabilities: dict[str, float],
) -> tuple[str, str]:
    system = """You are the Critic Agent in EmoAgent. The first image is the original source and the second is the edited result. Follow the paper's diagnosis order: whole-image target-emotion assessment, instruction emotional effectiveness, then execution accuracy. Return valid JSON only, without Markdown. Do not output hidden chain-of-thought; report concise visible evidence."""
    user = f"""Planning-Agent source-emotion prediction: {source_emotion_label}
Do not reclassify it during final result validation; report visible evidence if the edited transition is inconsistent.
Target emotion: {target_emotion}
Source analysis: {_dump(analysis)}
Executed plan: {_dump(plan)}
Step validation log: {_dump(step_results)}
Source ResNet probabilities (auxiliary metric evidence, not the source label): {_dump(source_probabilities)}
Edited ResNet probabilities (auxiliary evidence only): {_dump(edited_probabilities)}

Apply the paper's sequence:
1. Does the complete edited image effectively express the target emotion?
2. Which instructions do not contribute meaningfully and specifically to that target emotion?
3. Which emotionally valid instructions were not correctly executed?

Use visible objects, attributes, expressions, actions, background, lighting, and hue. A strong atmospheric treatment can contribute to emotion, especially when supported by semantic changes. Compare source and result at the level of recognizable main subject and broad scene coherence. Do not demand exact object counts, pixel geometry, lighting, background details, or spatial placement unless changing them destroys the subject's identity.

Reject an unrecognizable or replaced main subject, an unrelated scene, severe visual artifacts, implausible pasted content, or emotion-label text. Prominent additions are valid when they fit the scene and communicate the target emotion. Treat multiple_emotion_factors_present as true when either one strong cue or a coherent combination of cues makes the target emotion visible. If facial_expression_available is false, do not require facial-expression evidence; judge other cues instead.

Return:
{{
  "verdict": "correct or wrong",
  "perceived_emotion": "concise judgment grounded in visible evidence",
  "emotion_evidence": {{
    "objects": ["visible evidence"],
    "object_attributes": ["visible evidence"],
    "expressions_actions": ["visible evidence"],
    "background": ["visible evidence"],
    "global_hue_lighting": ["visible evidence"]
  }},
  "semantic_preservation": "good, acceptable, or poor",
  "facial_expression_constraint_obeyed": true,
  "main_subject_preserved": true,
  "key_information_preserved": true,
  "composition_preserved": true,
  "new_content_context_compatible": true,
  "multiple_emotion_factors_present": true,
  "brightness_and_detail_preserved": true,
  "instruction_diagnosis": [
    {{
      "step_id": 1,
      "emotional_effectiveness": "valid or ineffective",
      "execution_accuracy": "correct, incorrect, or not_applicable",
      "problem": "concise visible issue",
      "revised_instruction": "complete imperative correction, or empty"
    }}
  ],
  "updated_instructions": ["complete imperative instruction to re-apply"],
  "final_reason": "concise verifiable conclusion"
}}"""
    return system, user
