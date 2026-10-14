import unittest

from emoagent.pipeline import EmoAgentPipeline, PIPELINE_REVISION
from emoagent.prompts import flux_prompt_generation, plan_generation_prompt


class RelaxedPromptTests(unittest.TestCase):
    def test_revision_and_planner_allow_visible_emotional_edits(self):
        self.assertIn("first-prompt-relaxed", PIPELINE_REVISION)
        analysis = {
            "facial_expression_available": False,
            "protected_entities": [{"entity": "house"}],
            "must_preserve_layout": ["house remains recognizable"],
        }
        _, prompt = plan_generation_prompt(analysis, "contentment", "fear", [], 1, 4)
        self.assertIn("allow noticeable changes", prompt)
        self.assertIn("may become prominent", prompt)
        self.assertNotIn("must not cover, displace, shrink", prompt)

    def test_fallback_prompt_preserves_recognizability_not_exact_pixels(self):
        prompt = EmoAgentPipeline._controlled_edit_prompt(
            {"instruction": "Transform the sky into a dramatic storm."},
            {
                "target_region": "sky",
                "required_change": "a dramatic storm is visible",
                "placement_scale_depth": ["across the sky"],
                "must_remain_unchanged": ["main house identity"],
                "must_not_appear": ["text"],
            },
        )
        self.assertIn("main subject recognizable", prompt)
        self.assertIn("Allow natural changes", prompt)
        self.assertNotIn("Preserve every unmentioned source entity", prompt)

    def test_objective_gate_allows_dark_but_readable_edit(self):
        import tempfile
        from pathlib import Path
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, edited = root / "source.png", root / "edited.png"
            Image.new("RGB", (64, 64), (180, 180, 180)).save(source)
            Image.new("RGB", (64, 64), (30, 30, 30)).save(edited)
            self.assertTrue(EmoAgentPipeline._objective_image_checks(source, edited, "fear")["brightness_ok"])


if __name__ == "__main__":
    unittest.main()
